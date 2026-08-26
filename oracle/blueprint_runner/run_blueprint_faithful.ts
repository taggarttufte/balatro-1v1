/**
 * "Game-faithful" variant driver on top of Blueprint's Game class.
 *
 * Identical setup and streams to Blueprint's analyzeSeed(), with ONE behavioural addition the
 * published analyzers (Immolate / TheSoul / Blueprint / balatrohq) do not model:
 *
 *   card.lua Card:set_ability -> G.GAME.used_jokers[key] = true on EVERY card creation, and
 *   Card:remove clears it.  get_current_pool marks used keys UNAVAILABLE (resample).  Hence in
 *   the real game (a) slot 2 of a shop cannot repeat slot 1, and (b) a booster opened while the
 *   shop is displayed cannot contain a card currently shown in the shop.
 *
 * Policy encoded here (matches oracle/schema.md "shops"):
 *   per shop visit: draw slot 1, lock it, draw slot 2, lock it; choose both packs; open both
 *   packs (in order) with the two slots locked; unlock the slots.  After the ante's last visit,
 *   reroll pairs (lock first while drawing second, unlock after) until `cards` items exist.
 *
 *   cd oracle/blueprint_runner/vendor/Blueprint
 *   npx vite-node ../../run_blueprint_faithful.ts -- --seed-file ../../seeds.txt --antes 8 --cards 50 --out ../../_raw
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { EVENT_UNLOCKS, options as UNLOCK_OPTIONS } from "./vendor/Blueprint/src/modules/const.ts";
import { Game } from "./vendor/Blueprint/src/modules/balatrots/Game.ts";
import { Lock } from "./vendor/Blueprint/src/modules/balatrots/Lock.ts";
import { InstanceParams } from "./vendor/Blueprint/src/modules/balatrots/struct/InstanceParams.ts";
import { Deck, deckMap } from "./vendor/Blueprint/src/modules/balatrots/enum/Deck.ts";
import { Stake } from "./vendor/Blueprint/src/modules/balatrots/enum/Stake.ts";
import { Type } from "./vendor/Blueprint/src/modules/balatrots/enum/cards/CardType.ts";
import { JokerData } from "./vendor/Blueprint/src/modules/balatrots/struct/JokerData.ts";
import { Card } from "./vendor/Blueprint/src/modules/balatrots/enum/cards/Card.ts";
import type { StakeType } from "./vendor/Blueprint/src/modules/balatrots/enum/Stake.ts";

interface Args { seeds: string[]; antes: number; cards: number; deck: string; stake: string; version: number; out: string; }

function parseArgs(argv: string[]): Args {
    const a: Args = { seeds: [], antes: 8, cards: 50, deck: "Red Deck", stake: "White Stake", version: 10106, out: "./_raw" };
    for (let i = 0; i < argv.length; i++) {
        const k = argv[i], v = argv[i + 1];
        switch (k) {
            case "--seeds": a.seeds.push(...v.split(",").map(s => s.trim()).filter(Boolean)); i++; break;
            case "--seed-file":
                for (const line of fs.readFileSync(v, "utf8").split(/\r?\n/)) { const s = line.replace(/#.*$/, "").trim(); if (s) a.seeds.push(s); }
                i++; break;
            case "--antes": a.antes = parseInt(v, 10); i++; break;
            case "--cards": a.cards = parseInt(v, 10); i++; break;
            case "--deck": a.deck = v; i++; break;
            case "--stake": a.stake = v; i++; break;
            case "--version": a.version = parseInt(v, 10); i++; break;
            case "--out": a.out = v; i++; break;
            case "--": break;
            default: if (k.startsWith("--")) throw new Error(`unknown option ${k}`);
        }
    }
    if (!a.seeds.length) throw new Error("no seeds given");
    return a;
}

const sanitize = (s: string) => s.toUpperCase().replace(/0/g, "O").replace(/[^A-Z1-9]/g, "").slice(0, 8);

function makeEngine(seed: string, args: Args): Game {
    const deck = new Deck(deckMap[args.deck]);
    const params = new InstanceParams(deck, new Stake(args.stake as StakeType), false, args.version);
    const g = new Game(seed, params);
    g.hasSpoilers = false;
    g.initLocks(1, true, true);
    g.handleSelectedUnlocks([...UNLOCK_OPTIONS]);
    g.lockLevelTwoVouchers();
    g.lock(Array.from(Lock.firstLock));
    g.lock(Array.from(Lock.ante2Lock));
    g.setDeck(deck);
    EVENT_UNLOCKS.forEach((e: any) => g.lock(e.name));
    return g;
}

function itemOut(x: any): any {
    if (x instanceof JokerData) {
        return { type: "Joker", name: x.joker.getName(), edition: x.edition, rarity: x.rarity,
                 stickers: { eternal: x.stickers.eternal, perishable: x.stickers.perishable, rental: x.stickers.rental } };
    }
    if (x instanceof Card) {
        return { type: "Standard", base: x.getName(), enhancement: x.getEnhancement?.() ?? (x as any)._enhancement,
                 edition: ((x as any)._edition?.name) ?? "No Edition", seal: ((x as any)._seal?.name) ?? "No Seal" };
    }
    return { type: x.constructor?.name?.replace(/Item$|Enum$/, "") ?? "?", name: x.getName() };
}

/** Name used for used_jokers locking: jokers/consumables lock by display name; playing cards never do. */
function lockName(x: any): string | null {
    if (x instanceof JokerData) return x.joker.getName();
    if (x instanceof Card) return null;
    return x.getName();
}

function drawShopItem(g: Game, ante: number): { raw: any; out: any } {
    const si = g.nextShopItem(ante);
    const raw = si.type === Type.JOKER ? si.jokerData : si.item;
    const out = itemOut(raw);
    if (si.type === Type.TAROT) out.type = "Tarot";
    else if (si.type === Type.PLANET) out.type = "Planet";
    else if (si.type === Type.SPECTRAL) out.type = "Spectral";
    else if (si.type === Type.PLAYING_CARD) out.type = "Standard";
    if (out.name === "The Soul" || out.name === "Black Hole") out.type = "Spectral";
    return { raw, out };
}

function drawPair(g: Game, ante: number): { raws: any[]; outs: any[]; locked: string[] } {
    const a = drawShopItem(g, ante);
    const locked: string[] = [];
    const la = lockName(a.raw);
    if (la && !g.isLocked(la)) { g.lock(la); locked.push(la); }
    const b = drawShopItem(g, ante);
    const lb = lockName(b.raw);
    if (lb && !g.isLocked(lb)) { g.lock(lb); locked.push(lb); }
    return { raws: [a.raw, b.raw], outs: [a.out, b.out], locked };
}

function packOut(g: Game, ante: number) {
    const pt = g.nextPack(ante);
    const info = g.packInfo(pt);
    const cards = g.generatePack(info, ante).map((c: any) => {
        const o = itemOut(c);
        if (info.getKind() === "Arcana" && o.type !== "Spectral") o.type = "Tarot";
        if (info.getKind() === "Celestial") o.type = "Planet";
        if (info.getKind() === "Spectral") o.type = "Spectral";
        if (o.name === "The Soul" || o.name === "Black Hole") o.type = "Spectral";
        return o;
    });
    return { name: info.getKind(), size: info.getSize(), choices: info.getChoices(), cards };
}

function run(args: Args) {
    fs.mkdirSync(args.out, { recursive: true });
    for (const rawSeed of args.seeds) {
        const seed = sanitize(rawSeed);
        const g = makeEngine(seed, args);
        const antes: Record<number, any> = {};
        for (let ante = 1; ante <= args.antes; ante++) {
            g.initUnlocks(ante, false);
            const boss = g.nextBoss(ante).getName();
            const voucher = g.nextVoucher(ante).getName();
            const tags = [g.nextTag(ante).getName(), g.nextTag(ante).getName()];
            const queue: any[] = [];
            const visits = ante === 1 ? ["bigBlind", "bossBlind"] : ["smallBlind", "bigBlind", "bossBlind"];
            const blinds: Record<string, any> = { smallBlind: { packs: [] }, bigBlind: { packs: [] }, bossBlind: { packs: [] } };
            for (const v of visits) {
                const pair = drawPair(g, ante);
                queue.push(...pair.outs);
                blinds[v].packs.push(packOut(g, ante), packOut(g, ante));
                pair.locked.forEach(n => g.unlock(n));
            }
            while (queue.length < args.cards) {
                const pair = drawPair(g, ante);
                queue.push(...pair.outs);
                pair.locked.forEach(n => g.unlock(n));
            }
            antes[ante] = { boss, voucher, tags, queue: queue.slice(0, args.cards), blinds };
        }
        const out = { generator: "blueprint-faithful", settings: { seed, deck: args.deck, stake: args.stake, gameVersion: String(args.version), antes: args.antes, cardsPerAnte: args.cards },
                      generated_at: new Date().toISOString(), antes };
        fs.writeFileSync(path.join(args.out, `${seed}.faithful.json`), JSON.stringify(out));
        console.log(`${seed}: ante1 boss=${antes[1].boss} voucher=${antes[1].voucher}`);
    }
}

run(parseArgs(process.argv.slice(2)));
