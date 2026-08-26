/**
 * Re-run Blueprint's own accuracy fixtures (___tests___/seedJson/*.json, labelled
 * "immolateResults", i.e. produced by the original Immolate/TheSoul engine) and compare
 * field by field.  This documents that the vendored Blueprint commit still agrees with
 * Immolate on boss / voucher / tags / shop queue / packs for those 9 seeds.
 *
 *   cd oracle/blueprint_runner/vendor/Blueprint
 *   npx vite-node ../../check_fixtures.ts
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { analyzeSeed } from "./vendor/Blueprint/src/modules/ImmolateWrapper/index.ts";

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const fixtureDir = path.join(here, "vendor", "Blueprint", "___tests___", "seedJson");

function normBase(b: any): string | undefined {
    if (b === undefined || b === null) return undefined;
    return Array.isArray(b) ? b.join("") : String(b);
}

function cardSig(c: any): string {
    if (!c) return "<none>";
    const parts = [c.type, c.name];
    if (c.type === "Joker") parts.push(c.edition || "", String(c.rarity ?? ""), c.isEternal ? "E" : "", c.isPerishable ? "P" : "", c.isRental ? "R" : "");
    if (c.type === "Standard") parts.push(normBase(c.base) ?? "", c.enhancements ?? "", c.edition ?? "", c.seal ?? "");
    return parts.join("|");
}

let files = fs.readdirSync(fixtureDir).filter(f => f.endsWith(".json"));
let totalFields = 0, mismatches = 0;
const report: string[] = [];

for (const f of files) {
    const fx = JSON.parse(fs.readFileSync(path.join(fixtureDir, f), "utf8"));
    const st = fx.analyzeState;
    const got = analyzeSeed({
        seed: st.seed, deck: st.deck, stake: st.stake, gameVersion: st.gameVersion,
        antes: st.antes, cardsPerAnte: st.cardsPerAnte,
    }, fx.options) as any;
    const exp = fx.immolateResults;
    let fileMis = 0, fileFields = 0;
    const diffs: string[] = [];
    for (const anteKey of Object.keys(exp.antes)) {
        if (anteKey === "0") continue;
        const e = exp.antes[anteKey], g = got.antes[anteKey];
        const check = (label: string, ev: string, gv: string) => {
            fileFields++;
            if (ev !== gv) { fileMis++; if (diffs.length < 8) diffs.push(`  ante ${anteKey} ${label}: expected ${ev} got ${gv}`); }
        };
        check("boss", e.boss, g.boss);
        check("voucher", e.voucher, g.voucher);
        check("tags", e.tags.join(","), g.tags.join(","));
        for (let i = 0; i < e.queue.length; i++) check(`queue[${i}]`, cardSig(e.queue[i]), cardSig(g.queue[i]));
        for (const blind of ["smallBlind", "bigBlind", "bossBlind"]) {
            const ep = e.blinds[blind].packs, gp = g.blinds[blind].packs;
            check(`${blind}.npacks`, String(ep.length), String(gp.length));
            for (let p = 0; p < ep.length; p++) {
                check(`${blind}.pack[${p}]`, `${ep[p].name}/${ep[p].size}/${ep[p].choices}`, `${gp[p]?.name}/${gp[p]?.size}/${gp[p]?.choices}`);
                for (let c = 0; c < ep[p].cards.length; c++) check(`${blind}.pack[${p}].card[${c}]`, cardSig(ep[p].cards[c]), cardSig(gp[p]?.cards?.[c]));
            }
        }
    }
    totalFields += fileFields; mismatches += fileMis;
    report.push(`${st.seed.padEnd(9)} deck=${st.deck.padEnd(14)} stake=${st.stake.padEnd(12)} v=${st.gameVersion} antes=${st.antes}: ${fileFields - fileMis}/${fileFields} fields match${fileMis ? "  <-- MISMATCH" : ""}`);
    report.push(...diffs);
}
console.log(report.join("\n"));
console.log(`\nTOTAL: ${totalFields - mismatches}/${totalFields} fields match across ${files.length} fixtures`);
process.exit(mismatches ? 1 : 0);
