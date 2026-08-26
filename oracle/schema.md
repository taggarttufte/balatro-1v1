# Oracle ground-truth schema (`oracle/ground_truth/<SEED>.json`)

One file per seed. Keys are Balatro 1.0.1o internal keys (`G.P_CENTERS` / `G.P_BLINDS` /
`G.P_TAGS` / `G.P_CARDS`); every keyed object also carries the analyzer display `name`.
`oracle/keymap.py` is the name<->key table (with the analyzer aliases: "Speed Tag" = `tag_skip`,
"Canio" = `j_caino`, "Drivers License" = `j_drivers_license`, "Mail In Rebate" = `j_mail`).

Validate with `python -m oracle.parity_check --validate-only`.

```jsonc
{
  "schema_version": "1.0",
  "seed": "ALEEB",                       // as the game normalises it: upper-case, 0 -> O, [A-Z1-9], <= 8 chars
  "game_version": "1.0.1o",              // the game this oracle targets
  "analyzer_version_flag": "10106",      // Immolate-lineage version switch used (10106 = 1.0.1f+ pools)
  "deck": "b_red",        "deck_name": "Red Deck",
  "stake": "stake_white", "stake_name": "White Stake",
  "profile": "fully_unlocked",
  "assumptions": ["..."],                // prose list of modelling assumptions (see below)
  "shop_queue_depth": 50,                // items per ante in shop_queue
  "source": {
    "primary": "blueprint",
    "primary_detail": {"repo": "...", "commit": "<sha>", "driver": "oracle/blueprint_runner/run_blueprint.ts", "generated_at": "..."},
    "cross_checks": {
      "thesoul_wasm":  {"status": "agree" | "DISAGREE", "fields_compared": 2973, "mismatches": [...], ...},
      "balatrohq_ssr": {"status": "agree", "mismatches": []}          // ALEEB only
    }
  },
  "retrieved": "2026-08-21T17:00:00Z",

  "antes": {
    "1": {
      "boss":    {"key": "bl_window", "name": "The Window"},
      "voucher": {"key": "v_magic_trick", "name": "Magic Trick"},      // shown in every shop of this ante
      "tags":    {"small": {"key": "tag_skip", "name": "Speed Tag"},   // tag for skipping the Small Blind
                  "big":   {"key": "tag_skip", "name": "Speed Tag"}},  // tag for skipping the Big Blind
      "shop_queue": [ <item>, ... ],        // the ante's card stream, in draw order (see "shop_queue")
      "shops": [                            // shop visits of this ante, in order (2 at ante 1, else 3)
        {"visit": "after_small",            // "after_prev_boss" | "after_small" | "after_big"
         "analyzer_label": "bigBlind",      // what Immolate/TheSoul/Blueprint/balatrohq call it (blind you play NEXT)
         "packs": [ <pack>, <pack> ]}       // the two boosters offered at that visit, with contents
      ],
      "soul_spawns": [                      // every "The Soul" in this ante's packs
        {"shop": 0, "visit": "after_small", "pack": 1, "card": 2,
         "nth_soul_in_run": 1, "legendary_if_all_prior_souls_used": "j_caino"}
      ],
      "deck_order_unverified": {"note": "...", "small": ["S_T", ...52], "big": [...], "boss": [...]}
    },
    "2": { ... }, ... "8": { ... }
  },

  "legendary_stream": ["j_caino", "j_triboulet", "j_perkeo", ...],   // raw 'Joker4' stream, 5 deep
  "legendary_stream_note": "...",
  "first_soul_joker_by_ante": {"1": {"key": "j_caino", "edition": null}, ...},  // first Soul used at ante N: legendary + edisouN edition
  "voucher_chain_if_bought": {            // branch: buy the voucher every ante
    "note": "...",
    "antes": {"1": {"voucher": {...}, "after_buying": null, "shop_queue_first6_names": [...]},
              "2": {"voucher": {...}, "after_buying": "v_magic_trick", ...}, ...}
  },

  "variants": {
    "game_faithful_used_jokers": {        // see "Variants" below
      "note": "...", "driver": "oracle/blueprint_runner/run_blueprint_faithful.ts",
      "fields_differing": 3,
      "reasons": {"same_shop_duplicate": 1, "collides_with_displayed_shop": 2, "downstream_resample_shift": 0},
      "overrides": [{"path": "antes.4.shops[0].packs[1].cards[2]", "value": <item>}, ...]
    }
  }
}
```

## `<item>`

| set | fields |
|---|---|
| `Joker` | `key` (`j_*`), `name`, `edition` (`null`, `e_foil`, `e_holo`, `e_polychrome`, `e_negative`), `rarity` (1-4), `stickers` `{eternal, perishable, rental}` (all false at White stake) |
| `Tarot` / `Planet` / `Spectral` | `key` (`c_*`), `name`.  `c_soul` / `c_black_hole` are typed `Spectral`. |
| `Base` (playing card) | `key` = `G.P_CARDS` key `<S>_<R>` (`H_T` = 10 of Hearts; S in H/C/D/S, R in 2-9,T,J,Q,K,A), `name`, `enhancement` (`null` or `m_bonus`, `m_mult`, `m_wild`, `m_glass`, `m_steel`, `m_stone`, `m_gold`, `m_lucky`), `edition` (`null`, `e_foil`, `e_holo`, `e_polychrome`), `seal` (`null`, `Red`, `Blue`, `Gold`, `Purple` -- the game stores the bare colour) |

An item in the primary data that differs under the game-faithful variant additionally carries
`"analyzer_gap": "same_shop_duplicate" | "collides_with_displayed_shop" | "downstream_resample_shift"`.

## `<pack>`

```jsonc
{"key": "p_arcana_jumbo",     // art variant suffix (_1.._4) dropped: analyzers cannot resolve it
 "name": "Jumbo Arcana Pack", "kind": "Arcana", "size": 5, "choices": 1,
 "cards": [ <item> x size ]}  // contents when OPENED at that visit (see assumptions)
```

Pack key from shape: `choices == 2` -> mega; `size` 4/5 with `choices == 1` -> jumbo; else normal.
Sizes: Arcana/Celestial/Standard 3/5/5, Buffoon/Spectral 2/4/4 (normal/jumbo/mega).

## `shop_queue`

The per-ante stream the game draws shop cards from (`create_card_for_shop`: type roll `cdt{ante}`,
then `Joker{r}sho{ante}` / `Tarotsho{ante}` / `Planetsho{ante}` / `Spectralsho{ante}` / front `frontsho{ante}`,
edition `edisho{ante}`).  The game fills 2 slots per shop visit (`shop_joker_max`, +1 per Overstock)
and each reroll draws the next 2.  So with no purchases and no Overstock:

* items `[0:2]` = first visit of the ante, `[2:4]` = second, `[4:6]` = third (antes >= 2),
* items after that = what rerolls at the last visit would show, two at a time.

The stream is the same whichever visit you reroll at; only the pairing of items into visits moves.

## Shop visits and ante boundaries

`ease_ante(1)` runs when the Boss is beaten, *before* the post-boss shop, so that shop uses the
**new** ante's keys.  Hence ante 1 has two visits (after Small, after Big) and every later ante has
three: `after_prev_boss`, `after_small`, `after_big`.  The Immolate lineage labels visits by the blind
you are about to play (`smallBlind`/`bigBlind`/`bossBlind`), kept in `analyzer_label`.

The very first pack of the run is the forced Buffoon Pack (`get_pack`: `first_shop_buffoon`), which
consumes no `shop_pack1` RNG; every other pack is a weighted draw on `shop_pack{ante}` (weights sum 22.42).

## Assumptions (baked into the data)

1. Profile fully unlocked; fresh run: `enhancement_gate` jokers (Stone/Steel/Glass Joker, Golden Ticket,
   Lucky Cat), Cavendish (`yes_pool_flag`) and Planet X / Ceres / Eris (`softlock`) are UNAVAILABLE and get
   resampled past -- exactly what the game does with no such cards in the deck / hands never played.
2. No purchases, no Showman, no skips, no tags in effect, no vouchers redeemed (rates 20/4/4/0/0).
3. Every pack is opened at its visit, in order, before any reroll; pack N of a kind in an ante assumes
   packs 1..N-1 of that kind were opened (shared per-ante per-source streams such as `Tarotar1{ante}`).
4. `deck_order_unverified` is Blueprint's model of `G.deck:shuffle('nr'..ante)` and has no second source.
5. Bosses follow the game's least-used-first rule (`bosses_used`); ante % 8 == 0 draws from the five
   showdown bosses.  Tags with `min_ante = 2` (`tag_negative`, `tag_standard`, `tag_meteor`, `tag_buffoon`,
   `tag_handy`, `tag_garbage`, `tag_ethereal`, `tag_top_up`, `tag_orbital`) cannot appear at ante 1.

## Variants

`antes` is the **published-analyzer** output (Blueprint; independently reproduced bit-for-bit by TheSoul's
WASM build of the C++ Immolate on every field, and by balatrohq for ALEEB).  Those tools omit one
game rule: `Card:set_ability` marks *every* created card in `G.GAME.used_jokers` (cleared on
`Card:remove`), and `get_current_pool` marks used keys UNAVAILABLE.  Consequences in the real game:

* slot 2 of a shop can never repeat slot 1 (resample on `<pool>_resample2`...),
* a pack opened while the shop is displayed can never contain a displayed card,
* each such resample advances the shared `_resample{n}` streams, shifting later *analyzer* resamples too.

`variants.game_faithful_used_jokers` holds the full difference: apply `overrides` (JSON-path -> item)
to `antes` to get the used_jokers-faithful sequence.  Boss, voucher, tags and pack kinds never differ.
Across the 126-seed corpus (8 antes, 50-deep queues) 953 fields differ (222 same-shop duplicates,
319 pack/shop collisions, 412 downstream shifts); `python -m oracle.parity_check --variant faithful`
compares against that sequence.  See SOURCES.md for the evidence behind this variant.
