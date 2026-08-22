/**
 * Headless driver for TheSoul (SpectralPack/TheSoul): the Emscripten WASM build of
 * MathIsFun0's C++ Immolate.  This is an implementation independent of Blueprint's
 * TypeScript port, so agreement between the two is a meaningful cross-check.
 *
 *   node mp/oracle/blueprint_runner/run_thesoul.js --seeds A,B --antes 8 --cards 50 --out mp/oracle/blueprint_runner/_raw
 *
 * The per-ante loop mirrors TheSoul's index.html performAnalysis(), with ONE deliberate
 * difference: we call initLocks(1, false, true) (fresh run) instead of the web UI's
 * initLocks(1, false, false).  The game gates Stone/Steel/Glass Joker, Golden Ticket,
 * Lucky Cat (enhancement_gate), Cavendish (yes_pool_flag) and Planet X/Ceres/Eris
 * (softlock) out of a fresh run, so freshRun=true is the faithful setting and it matches
 * what Blueprint does.  Pass --web-ui-locks to reproduce the website's output instead.
 */
"use strict";
const fs = require("node:fs");
const path = require("node:path");

function parseArgs(argv) {
    const a = { seeds: [], antes: 8, cards: 50, deck: "Red Deck", stake: "White Stake", version: 10106, out: "./_raw", webUiLocks: false };
    for (let i = 0; i < argv.length; i++) {
        const k = argv[i], v = argv[i + 1];
        switch (k) {
            case "--seeds": a.seeds.push(...v.split(",").map(s => s.trim()).filter(Boolean)); i++; break;
            case "--seed-file":
                for (const line of fs.readFileSync(v, "utf8").split(/\r?\n/)) {
                    const s = line.replace(/#.*$/, "").trim();
                    if (s) a.seeds.push(s);
                }
                i++; break;
            case "--antes": a.antes = parseInt(v, 10); i++; break;
            case "--cards": a.cards = parseInt(v, 10); i++; break;
            case "--deck": a.deck = v; i++; break;
            case "--stake": a.stake = v; i++; break;
            case "--version": a.version = parseInt(v, 10); i++; break;
            case "--out": a.out = v; i++; break;
            case "--web-ui-locks": a.webUiLocks = true; break;
            default: if (k.startsWith("--")) throw new Error(`unknown option ${k}`);
        }
    }
    if (!a.seeds.length) throw new Error("no seeds given");
    return a;
}

const LEVEL2_VOUCHERS = ["Overstock Plus", "Liquidation", "Glow Up", "Reroll Glut", "Omen Globe", "Observatory",
    "Nacho Tong", "Recyclomancy", "Tarot Tycoon", "Planet Tycoon", "Money Tree", "Antimatter", "Illusion",
    "Petroglyph", "Retcon", "Palette"];

function sanitize(seed) {
    return seed.toUpperCase().replace(/0/g, "O").replace(/[^A-Z1-9]/g, "").slice(0, 8);
}

function vecToArray(vec) {
    const out = [];
    for (let i = 0; i < vec.size(); i++) out.push(vec.get(i));
    return out;
}

function analyze(Immolate, args, seed) {
    const inst = new Immolate.Instance(seed);
    inst.params = new Immolate.InstParams(args.deck, args.stake, false, args.version);
    inst.initLocks(1, false, !args.webUiLocks);
    for (const v of LEVEL2_VOUCHERS) inst.lock(v);
    // fully unlocked profile: nothing else locked
    inst.setStake(args.stake);
    inst.setDeck(args.deck);

    const antes = {};
    for (let a = 1; a <= args.antes; a++) {
        inst.initUnlocks(a, false);
        const ante = { boss: inst.nextBoss(a) };
        const voucher = inst.nextVoucher(a);
        ante.voucher = voucher;
        // TheSoul's UI locks the shown voucher for the rest of the run and unlocks its
        // level 2.  That models "you bought it" for later antes.  We do NOT do that here,
        // to match the no-purchase baseline; the voucher can therefore recur in ante N+1.
        ante.tags = [inst.nextTag(a), inst.nextTag(a)];
        ante.queue = [];
        for (let q = 0; q < args.cards; q++) {
            const item = inst.nextShopItem(a);
            const entry = { type: item.type, name: item.item };
            if (item.type === "Joker") {
                const jd = item.jokerData;
                entry.edition = jd.edition;
                entry.rarity = jd.rarity;
                entry.stickers = { eternal: jd.stickers.eternal, perishable: jd.stickers.perishable, rental: jd.stickers.rental };
            }
            ante.queue.push(entry);
            item.delete();
        }
        const numPacks = a === 1 ? 4 : 6;
        ante.packs = [];
        for (let p = 0; p < numPacks; p++) {
            const packName = inst.nextPack(a);
            const info = Immolate.packInfo(packName);
            const pack = { name: packName, kind: info.type, size: info.size, choices: info.choices, cards: [] };
            let cards;
            switch (info.type) {
                case "Celestial Pack": cards = inst.nextCelestialPack(info.size, a); pack.cards = vecToArray(cards).map(n => ({ type: "Planet", name: n })); break;
                case "Arcana Pack": cards = inst.nextArcanaPack(info.size, a); pack.cards = vecToArray(cards).map(n => ({ type: "Tarot", name: n })); break;
                case "Spectral Pack": cards = inst.nextSpectralPack(info.size, a); pack.cards = vecToArray(cards).map(n => ({ type: "Spectral", name: n })); break;
                case "Buffoon Pack": {
                    cards = inst.nextBuffoonPack(info.size, a);
                    for (let c = 0; c < info.size; c++) {
                        const j = cards.get(c);
                        pack.cards.push({ type: "Joker", name: j.joker, edition: j.edition, rarity: j.rarity,
                            stickers: { eternal: j.stickers.eternal, perishable: j.stickers.perishable, rental: j.stickers.rental } });
                        j.delete();
                    }
                    break;
                }
                case "Standard Pack": {
                    cards = inst.nextStandardPack(info.size, a);
                    for (let c = 0; c < info.size; c++) {
                        const cd = cards.get(c);
                        pack.cards.push({ type: "Standard", base: cd.base, enhancement: cd.enhancement, edition: cd.edition, seal: cd.seal });
                        cd.delete();
                    }
                    break;
                }
                default: throw new Error(`unknown pack kind ${info.type}`);
            }
            if (cards) cards.delete();
            info.delete();
            ante.packs.push(pack);
        }
        antes[a] = ante;
    }
    inst.delete();
    return antes;
}

function main() {
    const args = parseArgs(process.argv.slice(2));
    const soulDir = path.join(__dirname, "vendor", "TheSoul");
    // immolate.js is a non-modularized Emscripten build: it reads a pre-existing global
    // `Immolate` object for config and exports the Module via module.exports.
    global.Immolate = {
        locateFile: (f) => path.join(soulDir, f),
        onRuntimeInitialized() {
            fs.mkdirSync(args.out, { recursive: true });
            for (const raw of args.seeds) {
                const seed = sanitize(raw);
                const t0 = Date.now();
                const antes = analyze(Immolate, args, seed);
                const out = {
                    generator: "thesoul-wasm",
                    settings: { seed, deck: args.deck, stake: args.stake, gameVersion: String(args.version), antes: args.antes, cardsPerAnte: args.cards },
                    fresh_run_locks: !args.webUiLocks,
                    generated_at: new Date().toISOString(),
                    elapsed_ms: Date.now() - t0,
                    antes,
                };
                const file = path.join(args.out, `${seed}.thesoul.json`);
                fs.writeFileSync(file, JSON.stringify(out));
                console.log(`${seed}: ante1 boss=${antes[1].boss} voucher=${antes[1].voucher} tags=${antes[1].tags.join("/")} -> ${file}`);
            }
        },
    };
    require(path.join(soulDir, "immolate.js"));
}

main();
