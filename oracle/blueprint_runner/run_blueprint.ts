/**
 * Headless driver for Blueprint's seed analyzer (miaklwalker/Blueprint, a TypeScript
 * port of Immolate / TheSoul).  Emits one raw JSON per seed; mp/oracle/build_ground_truth.py
 * converts those into the mp/oracle/schema.md format.
 *
 * Run from the Blueprint checkout so vite-node picks up its tsconfig:
 *
 *   cd mp/oracle/blueprint_runner/vendor/Blueprint
 *   npx vite-node ../../run_blueprint.ts -- --seeds 7LB2WVPK,ALEEB --antes 8 --out ../../_raw
 *
 * Options (all optional):
 *   --seeds A,B,C        comma-separated seeds (0 is normalised to O, like the game)
 *   --seed-file path     one seed per line (# comments allowed)
 *   --antes N            antes to generate (default 8)
 *   --cards N            shop-queue depth per ante (default 50)
 *   --deck "Red Deck"    Blueprint deck name (default "Red Deck")
 *   --stake "White Stake"
 *   --version 10106      10106 = 1.0.1f+ pools (what the community uses for 1.0.1o)
 *   --out dir            output directory (default ./_raw)
 *   --buy-vouchers       also run the "buy every voucher" branch and record the chain
 *   --no-unlock-all      keep Blueprint's fresh-profile locks (default: fully unlocked profile)
 *
 * Modelling assumptions inherited from Blueprint (see SOURCES.md):
 *   - fresh profile with every unlockable unlocked (unless --no-unlock-all)
 *   - fresh run: Stone/Steel/Glass Joker, Golden Ticket, Lucky Cat, Cavendish, Planet X,
 *     Ceres, Eris are unavailable (matches the game's enhancement_gate / pool_flag / softlock)
 *   - no purchases, no rerolls, no Showman
 *   - every booster pack is opened, in display order
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { analyzeSeed } from "./vendor/Blueprint/src/modules/ImmolateWrapper/index.ts";
import { options as UNLOCK_OPTIONS } from "./vendor/Blueprint/src/modules/const.ts";
import { Game } from "./vendor/Blueprint/src/modules/balatrots/Game.ts";
import { InstanceParams } from "./vendor/Blueprint/src/modules/balatrots/struct/InstanceParams.ts";
import { Deck, deckMap } from "./vendor/Blueprint/src/modules/balatrots/enum/Deck.ts";
import { Stake } from "./vendor/Blueprint/src/modules/balatrots/enum/Stake.ts";
import { RNGSource } from "./vendor/Blueprint/src/modules/balatrots/enum/QueueName.ts";
import type { StakeType } from "./vendor/Blueprint/src/modules/balatrots/enum/Stake.ts";

interface Args {
    seeds: string[];
    antes: number;
    cards: number;
    deck: string;
    stake: string;
    version: string;
    out: string;
    buyVouchers: boolean;
    unlockAll: boolean;
}

function parseArgs(argv: string[]): Args {
    const a: Args = {
        seeds: [],
        antes: 8,
        cards: 50,
        deck: "Red Deck",
        stake: "White Stake",
        version: "10106",
        out: "./_raw",
        buyVouchers: false,
        unlockAll: true,
    };
    for (let i = 0; i < argv.length; i++) {
        const k = argv[i];
        const v = argv[i + 1];
        switch (k) {
            case "--seeds": a.seeds.push(...v.split(",").map(s => s.trim()).filter(Boolean)); i++; break;
            case "--seed-file": {
                const txt = fs.readFileSync(v, "utf8");
                for (const line of txt.split(/\r?\n/)) {
                    const s = line.replace(/#.*$/, "").trim();
                    if (s) a.seeds.push(s);
                }
                i++; break;
            }
            case "--antes": a.antes = parseInt(v, 10); i++; break;
            case "--cards": a.cards = parseInt(v, 10); i++; break;
            case "--deck": a.deck = v; i++; break;
            case "--stake": a.stake = v; i++; break;
            case "--version": a.version = v; i++; break;
            case "--out": a.out = v; i++; break;
            case "--buy-vouchers": a.buyVouchers = true; break;
            case "--no-unlock-all": a.unlockAll = false; break;
            case "--": break;
            default:
                if (k.startsWith("--")) throw new Error(`unknown option ${k}`);
        }
    }
    if (a.seeds.length === 0) throw new Error("no seeds given (--seeds or --seed-file)");
    return a;
}

function sanitize(seed: string): string {
    return seed.toUpperCase().replace(/0/g, "O").replace(/[^A-Z1-9]/g, "").slice(0, 8);
}

function baseOptions(unlockAll: boolean) {
    return {
        buys: {} as Record<string, any>,
        sells: {} as Record<string, any>,
        showCardSpoilers: false,
        unlocks: unlockAll ? [...UNLOCK_OPTIONS] : [],
        events: [] as any[],
        updates: [] as any[],
        lockedCards: {},
        maxMiscCardSource: 15,
    };
}

/** Legendary ("Joker4") stream and the per-ante edition a Soul-spawned joker would get. */
function legendaryInfo(seed: string, deck: string, stake: string, version: number, antes: number) {
    const mk = () => new Game(seed, new InstanceParams(new Deck(deckMap[deck]), new Stake(stake as StakeType), false, version));
    // Raw Joker4 stream: no purchase-locking, so repeats are possible (the game would
    // resample a legendary you already own unless Showman is held).
    const g = mk();
    const stream: string[] = [];
    for (let i = 0; i < 5; i++) {
        stream.push(g.nextJoker(RNGSource.S_Soul, 1, false).joker.getName());
    }
    // First Soul opened at ante A: legendary = first Joker4 draw, edition from edisou{A}.
    const firstByAnte: Record<number, { name: string; edition: string }> = {};
    for (let a = 1; a <= antes; a++) {
        const jd = mk().nextJoker(RNGSource.S_Soul, a, false);
        firstByAnte[a] = { name: jd.joker.getName(), edition: jd.edition };
    }
    return { legendary_stream: stream, first_soul_joker_by_ante: firstByAnte };
}

function stripMisc(result: any) {
    // Misc card sources are Blueprint-specific "what-if" streams whose state depends on the
    // analyzer's own consumption order.  Keep only the names so the raw file stays small.
    for (const ante of Object.values(result.antes) as any[]) {
        if (ante?.miscCardSources) {
            ante.miscCardSources = ante.miscCardSources.map((s: any) => ({
                name: s.name,
                source: s.source,
                cardType: s.cardType,
                cards: s.cards.map((c: any) => ({
                    name: c.name, type: c.type, edition: c.edition ?? undefined,
                    rarity: c.rarity ?? undefined, base: c.base ?? undefined,
                    enhancements: c.enhancements ?? undefined, seal: c.seal ?? undefined,
                })),
            }));
        }
    }
    return result;
}

function run(args: Args) {
    fs.mkdirSync(args.out, { recursive: true });
    const version = parseInt(args.version, 10);
    for (const rawSeed of args.seeds) {
        const seed = sanitize(rawSeed);
        const settings = {
            seed,
            deck: args.deck,
            stake: args.stake,
            gameVersion: args.version,
            antes: args.antes,
            cardsPerAnte: args.cards,
        };
        const t0 = Date.now();
        const main = analyzeSeed(settings, baseOptions(args.unlockAll) as any) as any;
        if (!main) throw new Error(`analyzeSeed returned nothing for ${rawSeed}`);
        delete main.antes[0]; // Blueprint's synthetic "ante 0" (pre-run) entry

        let voucherChain: Record<string, any> | undefined;
        if (args.buyVouchers) {
            // Buy the voucher in every ante: ante N's buy key is `${N}-VOUCHER-0`.
            // Blueprint consumes the buy after generating ante N, so the chain shows what
            // ante N+1 offers once N's voucher is used (and its level-2 unlocked).
            const opts = baseOptions(args.unlockAll);
            const chain: Record<string, any> = {};
            let prev: string | undefined;
            for (let a = 1; a <= args.antes; a++) {
                const r = analyzeSeed({ ...settings, antes: a }, opts as any) as any;
                const v = r.antes[a].voucher;
                chain[a] = { voucher: v, after_buying: prev ?? null, shop_queue_first6: r.antes[a].queue.slice(0, 6).map((c: any) => c.name) };
                opts.buys[`${a}-VOUCHER-0`] = { name: v };
                prev = v;
            }
            voucherChain = chain;
        }

        const out = {
            generator: "blueprint",
            blueprint_commit: process.env.BLUEPRINT_COMMIT ?? null,
            settings,
            unlock_all: args.unlockAll,
            generated_at: new Date().toISOString(),
            elapsed_ms: Date.now() - t0,
            result: stripMisc(main),
            legendary: legendaryInfo(seed, args.deck, args.stake, version, args.antes),
            voucher_chain_if_bought: voucherChain ?? null,
        };
        const file = path.join(args.out, `${seed}.blueprint.json`);
        fs.writeFileSync(file, JSON.stringify(out));
        console.log(`${seed}: ante1 boss=${main.antes[1].boss} voucher=${main.antes[1].voucher} tags=${main.antes[1].tags.join("/")} -> ${file}`);
    }
}

run(parseArgs(process.argv.slice(2)));
