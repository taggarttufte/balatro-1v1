"""Every pseudoseed / pseudorandom key used by Balatro 1.0.1o, with its construction rule.

Inventory built by grepping ``pseudoseed(``, ``pseudorandom(``, ``pseudorandom_element(`` and
``pseudoshuffle(`` across ALL 32 Lua files inside Balatro.exe (1.0.1o-FULL), not just the
subset extracted to ``mp/_reference/balatro_src/``.  Sites in ``functions/UI_definitions.lua``
are NOT in that extracted subset (see NOTES_POOLS.md, "Reference extraction gap").

How a key becomes a number (misc_functions.lua:297-319)
-------------------------------------------------------
``pseudoseed(key)`` keeps one LCG state per *distinct key string* in ``G.GAME.pseudorandom``:
first use seeds it with ``pseudohash(key .. seed)``; every use steps it
``abs(fmt('%.13f', (2.134453429141 + x*1.72431234) % 1))`` and returns
``(x + pseudohash(seed)) / 2``.  ``pseudorandom(k)`` then does ``math.randomseed(v);
math.random()`` (or ``math.random(min, max)``).  Therefore:

* the ante suffix, the ``key_append`` and the ``_resample<n>`` suffix each create an
  INDEPENDENT stream (``'Joker1sho1'``, ``'Joker1buf1'``, ``'Joker1sho1_resample2'`` ...);
* the same literal used at several sites shares ONE stream (``'8ba'`` is both 8 Ball and the
  Purple Seal; ``'wheel_of_fortune'`` is rolled three times in a row by one Wheel use);
* the order of calls within a stream matters, the order between streams does not.

``pseudorandom_element(t, seed)`` (misc_functions.lua:253-267) picks ``keys[math.random(#keys)]``
where ``keys`` is ``pairs(t)`` sorted: by ``v.sort_id`` if the first value is a table with one
(Card objects -> creation order), else by key (array pools -> index order; hash tables such as
``G.P_CARDS`` and the boss table -> byte-wise key-string order).
``pseudoshuffle(list, seed)`` (misc_functions.lua:206-217) first sorts by ``sort_id`` when
present, then runs Fisher-Yates ``for i = #list, 2, -1 do j = math.random(i); swap(i, j)``.

Record fields
-------------
name        short identifier
pattern     Python ``str.format`` pattern for the key ({ante}, {append}, {rarity}, {type}, {it})
lua         the literal construction expression in the source
primitive   pseudorandom | pseudorandom_int | pseudorandom_element | pseudoshuffle
via         wrapper function, if any
site        file.lua:line (paths relative to the game archive root)
event       what the draw decides
ante        True if G.GAME.round_resets.ante is concatenated
append      tuple of key_append values that can appear, or ()
resample    True if ``'_resample'..it`` (it = 2, 3, ...) is appended on UNAVAILABLE
pool        for element/shuffle draws: what is drawn from and its effective ordering
note        extra semantics
"""

# ---------------------------------------------------------------------------------------
# Key-construction helpers (pure string functions; mirror the Lua concatenations exactly)
# ---------------------------------------------------------------------------------------

def rarity_key(ante, append=""):
    """common_events.lua:1969  'rarity'..ante..(append or '')"""
    return f"rarity{ante}{append}"


def pool_key(pool_type, ante, append="", rarity=None, legendary=False):
    """common_events.lua:1971-1973 + :2052.

    Joker:   'Joker'..rarity..((not legendary and append) or '')  then ..ante unless legendary
             -> 'Joker1sho1', 'Joker3rta2', legendary -> 'Joker4' (no append, NO ante)
    others:  type..(append or '')..ante   -> 'Tarotar1', 'Planetpl12', 'Voucher1', 'Tag3',
             'Enhancedsta1', 'Spectralspe4'
    """
    if pool_type == "Joker":
        if legendary:
            return "Joker4"
        return f"Joker{rarity}{append}{ante}"
    return f"{pool_type}{append}{ante}"


def resample_key(base_key, it):
    """common_events.lua:1908/1921/2118  _pool_key..'_resample'..it   (it starts at 2)."""
    return f"{base_key}_resample{it}"


def rarity_from_poll(r):
    """common_events.lua:1970  r > 0.95 -> 3, r > 0.7 -> 2, else 1."""
    return 3 if r > 0.95 else 2 if r > 0.7 else 1


# key_append values passed to create_card(), with the event that passes each one.
KEY_APPENDS = {
    "sho":   "shop slot (create_card_for_shop, UI_definitions.lua:776) -- Joker/Tarot/Planet/Spectral/Base/Enhanced",
    "buf":   "Buffoon pack card (card.lua:1774)",
    "ar1":   "Arcana pack Tarot card (card.lua:1734)",
    "ar2":   "Arcana pack Spectral card via Omen Globe (card.lua:1732)",
    "pl1":   "Celestial pack Planet card (card.lua:1752/1754; forced key when Telescope and i==1)",
    "spe":   "Spectral pack card (card.lua:1757)",
    "sta":   "Standard pack playing card, Base or Enhanced (card.lua:1759)",
    "jud":   "Judgement tarot -> Joker (card.lua:1418)",
    "sou":   "The Soul -> legendary Joker (card.lua:1418); append is DROPPED from the pool key",
    "wra":   "Wraith -> Joker with _rarity=0.99 (Rare) (card.lua:1457)",
    "emp":   "The Emperor -> Tarot (card.lua:1406)",
    "pri":   "The High Priestess -> Planet (card.lua:1406)",
    "rif":   "Riff-raff -> Joker with _rarity=0 (Common), 2 jokers (card.lua:2535)",
    "top":   "Top-up Tag -> Joker with _rarity=0 (Common), 2 jokers (tag.lua:138)",
    "rta":   "Rare Tag -> Joker with _rarity=1 (Rare) (tag.lua:356)",
    "uta":   "Uncommon Tag -> Joker with _rarity=0.9 (Uncommon) (tag.lua:370)",
    "8ba":   "8 Ball (card.lua:3115) AND Purple Seal (card.lua:2260) -> Tarot; SAME stream",
    "hal":   "Hallucination -> Tarot (card.lua:2343)",
    "car":   "Cartomancer -> Tarot (card.lua:2551)",
    "sixth": "Sixth Sense -> Spectral (card.lua:2611)",
    "vag":   "Vagabond -> Tarot (card.lua:3750)",
    "sup":   "Superposition -> Tarot (card.lua:3774)",
    "sea":   "Seance -> Spectral (card.lua:3794)",
    "fool":  "The Fool -> forced key G.GAME.last_tarot_planet (card.lua:1377); NO pool draw",
    "blusl": "Blue Seal -> forced Planet for last hand (card.lua:1054); NO pool draw",
    "deck":  "Magic/Ghost deck starting consumables, forced key (back.lua:189); NO pool draw",
}

# Fixed-rarity create_card callers: _rarity argument -> rarity pool (no 'rarity..' roll consumed)
FORCED_RARITY = {"wra": 3, "rif": 1, "top": 1, "rta": 3, "uta": 2}


def _k(name, pattern, lua, primitive, site, event, *, via=None, ante=False, append=(),
       resample=False, pool=None, note=None):
    return {"name": name, "pattern": pattern, "lua": lua, "primitive": primitive, "via": via,
            "site": site, "event": event, "ante": ante, "append": tuple(append),
            "resample": resample, "pool": pool, "note": note}


KEYS = [
    # ------------------------------------------------------------------ pool draws (get_current_pool)
    _k("rarity_roll", "rarity{ante}{append}", "'rarity'..G.GAME.round_resets.ante..(_append or '')",
       "pseudorandom", "functions/common_events.lua:1969", "joker rarity for a pool draw",
       via="get_current_pool", ante=True, append=("sho", "buf", "jud", "sou"),
       note="Only rolled when _rarity is nil: shop ('sho'), Buffoon pack ('buf'), Judgement ('jud') and -- "
            "wastefully -- The Soul ('sou', result overridden to 4). Thresholds >0.95 Rare, >0.7 Uncommon."),
    _k("joker_pool", "Joker{rarity}{append}{ante}", "'Joker'..rarity..((not _legendary and _append) or '')..ante",
       "pseudorandom_element", "functions/common_events.lua:2114", "which joker (shop, pack, tarot, tag)",
       via="create_card/get_current_pool", ante=True,
       append=("sho", "buf", "jud", "wra", "rif", "top", "rta", "uta"), resample=True,
       pool="P_JOKER_RARITY_POOLS[rarity] (pools.JOKERS_BY_RARITY) with ineligible entries replaced by UNAVAILABLE in place"),
    _k("legendary_pool", "Joker4", "'Joker'..4  (append and ante both dropped when _legendary)",
       "pseudorandom_element", "functions/common_events.lua:2114", "which legendary (The Soul)",
       via="create_card/get_current_pool", ante=False, append=("sou",), resample=True,
       pool="P_JOKER_RARITY_POOLS[4] = caino, triboulet, yorick, chicot, perkeo",
       note="Stateless predictor get_first_legendary(seed) (misc_functions.lua:249) = pseudoseed('Joker4', seed) equals the in-run first draw."),
    _k("tarot_pool", "Tarot{append}{ante}", "'Tarot'..(_append or '')..ante",
       "pseudorandom_element", "functions/common_events.lua:2114", "which tarot",
       via="create_card/get_current_pool", ante=True,
       append=("sho", "ar1", "emp", "8ba", "hal", "car", "vag", "sup"), resample=True,
       pool="pools.TAROTS (22) with UNAVAILABLE in place (used_jokers w/o Showman, banned)"),
    _k("planet_pool", "Planet{append}{ante}", "'Planet'..(_append or '')..ante",
       "pseudorandom_element", "functions/common_events.lua:2114", "which planet",
       via="create_card/get_current_pool", ante=True, append=("sho", "pl1", "pri"), resample=True,
       pool="pools.PLANETS (12); softlock planets UNAVAILABLE until their hand was played"),
    _k("spectral_pool", "Spectral{append}{ante}", "'Spectral'..(_append or '')..ante",
       "pseudorandom_element", "functions/common_events.lua:2114", "which spectral",
       via="create_card/get_current_pool", ante=True, append=("sho", "ar2", "spe", "sixth", "sea"), resample=True,
       pool="pools.SPECTRALS (18); c_soul and c_black_hole ALWAYS UNAVAILABLE (18 slots, 16 live)"),
    _k("enhanced_pool", "Enhanced{append}{ante}", "'Enhanced'..(_append or '')..ante",
       "pseudorandom_element", "functions/common_events.lua:2114", "enhancement of a shop/pack playing card",
       via="create_card/get_current_pool", ante=True, append=("sho", "sta"), resample=False,
       pool="pools.ENHANCEMENTS (8, m_bonus..m_lucky); nothing is ever UNAVAILABLE"),
    _k("voucher_pool", "Voucher{ante}", "'Voucher'..ante",
       "pseudorandom_element", "functions/common_events.lua:1904", "shop voucher for the ante (rolled at run start and after each boss)",
       via="get_next_voucher_key/get_current_pool", ante=True, resample=True,
       pool="pools.VOUCHERS (32, base/plus interleaved); redeemed, unmet-requires, locked and already-displayed vouchers UNAVAILABLE"),
    _k("voucher_pool_fromtag", "Voucher_fromtag", "'Voucher_fromtag'  (REPLACES the whole key: no ante)",
       "pseudorandom_element", "functions/common_events.lua:1902-1908", "Voucher Tag extra voucher",
       via="get_next_voucher_key(true)", ante=False, resample=True,
       pool="same culled voucher pool as voucher_pool"),
    _k("tag_pool", "Tag{ante}", "'Tag'..(append or '')..ante  (append is never passed in vanilla)",
       "pseudorandom_element", "functions/common_events.lua:1917", "Small/Big blind skip tags (2 draws per ante: Small then Big)",
       via="get_next_tag_key/get_current_pool", ante=True, resample=True,
       pool="pools.TAGS (24); UNAVAILABLE if requires-center undiscovered or min_ante > ante; empty -> tag_handy"),
    _k("resample", "{base}_resample{it}", "_pool_key..'_resample'..it   (it = 2, 3, ...)",
       "pseudorandom_element", "functions/common_events.lua:1908,1921,2118", "redraw after hitting an UNAVAILABLE slot",
       via="create_card/get_next_voucher_key/get_next_tag_key", resample=True,
       note="Each retry is a NEW stream keyed by the suffix; the pool is not shrunk."),

    # ------------------------------------------------------------------ create_card extras
    _k("soul_roll", "soul_{type}{ante}", "'soul_'.._type..ante",
       "pseudorandom", "functions/common_events.lua:2091,2097", "0.3 % forced The Soul / Black Hole in packs",
       via="create_card (soulable=true only: booster packs)", ante=True,
       note="Tarot/Spectral/Tarot_Planet roll for c_soul; Planet/Spectral roll for c_black_hole. Spectral rolls the "
            "SAME key twice in sequence (black hole second, overrides). Threshold > 0.997. Skipped if that card is in used_jokers without Showman."),
    _k("front", "front{append}{ante}", "'front'..(key_append or '')..ante",
       "pseudorandom_element", "functions/common_events.lua:2124", "rank+suit of a Base/Enhanced shop or pack card",
       via="create_card", ante=True, append=("sho", "sta"),
       pool="G.P_CARDS hash -> key-string order (pools.PLAYING_CARD_KEYS: C_2..C_9,C_A,C_J,C_K,C_Q,C_T,D_...,H_...,S_...)"),
    _k("front_bare", "front", "'front'", "pseudorandom_element", "functions/common_events.lua:1929",
       "default front in create_playing_card when none supplied", via="create_playing_card",
       pool="G.P_CARDS key-string order", note="No vanilla caller reaches this default (both callers pass front)."),
    _k("edition_shop", "edi{append}{ante}", "'edi'..(key_append or '')..ante",
       "pseudorandom", "functions/common_events.lua:2150", "edition of every Joker created by create_card",
       via="poll_edition", ante=True, append=("sho", "buf", "jud", "sou", "wra", "rif", "top", "rta", "uta"),
       note="poll_edition thresholds: negative 0.003 (not scaled), polychrome 0.006*edition_rate, holo 0.02*er, foil 0.04*er."),
    _k("eternal_perishable", "etperpoll{ante} | packetper{ante}", "(area == G.pack_cards and 'packetper' or 'etperpoll')..ante",
       "pseudorandom", "functions/common_events.lua:2138", "eternal (>0.7, stake>=4) / perishable (0.4-0.7, stake>=7) sticker on shop/pack jokers",
       via="create_card", ante=True, note="Rolled for every shop/pack Joker regardless of stake (stream advances even at white stake)."),
    _k("rental", "ssjr{ante} | packssjr{ante}", "(area == G.pack_cards and 'packssjr' or 'ssjr')..ante",
       "pseudorandom", "functions/common_events.lua:2144", "rental sticker (>0.7)",
       via="create_card", ante=True, note="Only rolled when G.GAME.modifiers.enable_rentals_in_shop (gold stake)."),

    # ------------------------------------------------------------------ shop / packs
    _k("shop_type", "cdt{ante}", "'cdt'..G.GAME.round_resets.ante",
       "pseudorandom", "functions/UI_definitions.lua:766", "card type of each shop slot (Joker/Tarot/Planet/Base|Enhanced/Spectral by rate walk)",
       via="create_card_for_shop", ante=True,
       note="total_rate = joker_rate(20)+tarot_rate(4)+planet_rate(4)+playing_card_rate(0)+spectral_rate(0). NOT EXTRACTED file."),
    _k("illusion", "illusion", "'illusion'", "pseudorandom", "functions/UI_definitions.lua:772,786,787",
       "Illusion voucher: (a) >0.6 Enhanced vs Base for the playing-card slot type -- rolled while BUILDING the type list, "
       "i.e. every shop slot when Illusion is owned, before the cdt walk; (b) >0.8 edition trigger; (c) edition: >0.85 poly, >0.5 holo, else foil",
       via="create_card_for_shop", note="One stream, up to three draws per playing-card slot. NOT EXTRACTED file."),
    _k("shop_pack", "shop_pack{ante}", "(_key or 'pack_generic')..ante",
       "pseudorandom", "functions/common_events.lua:1953", "which booster in each of the 2 pack slots (weighted walk over pools.BOOSTERS)",
       via="get_pack('shop_pack') game.lua:3148", ante=True,
       note="First pack slot of a run is forced to p_buffoon_normal_<math.random(1,2)> WITHOUT consuming this stream (first_shop_buffoon)."),
    _k("omen_globe", "omen_globe", "'omen_globe'", "pseudorandom", "card.lua:1731",
       "Arcana pack card becomes Spectral (>0.8) when Omen Globe owned", via="Card:open"),
    _k("std_set", "stdset{ante}", "'stdset'..G.GAME.round_resets.ante", "pseudorandom", "card.lua:1759",
       "Standard pack card: Enhanced if >0.6 else Base", via="Card:open", ante=True),
    _k("std_edition", "standard_edition{ante}", "'standard_edition'..G.GAME.round_resets.ante", "pseudorandom", "card.lua:1761",
       "Standard pack card edition, poll_edition(mod=2, no_neg)", via="poll_edition", ante=True),
    _k("std_seal", "stdseal{ante}", "'stdseal'..G.GAME.round_resets.ante", "pseudorandom", "card.lua:1764",
       "Standard pack card gets a seal if > 0.8", via="Card:open", ante=True),
    _k("std_seal_type", "stdsealtype{ante}", "'stdsealtype'..G.GAME.round_resets.ante", "pseudorandom", "card.lua:1766",
       "seal type: >0.75 Red, >0.5 Blue, >0.25 Gold, else Purple", via="Card:open", ante=True),

    # ------------------------------------------------------------------ run / round structure
    _k("boss", "boss", "'boss'", "pseudorandom_element", "functions/common_events.lua:2379",
       "boss blind for the ante (run start, then every boss defeat; also Boss Tag / Director's Cut rerolls)",
       via="get_new_boss",
       pool="HASH table of eligible boss keys -> byte-wise alphabetical by key (pools.BOSS_KEYS_ALPHA filtered by eligibility and min bosses_used)"),
    _k("shuffle_start", "shuffle", "pseudoseed(_seed or 'shuffle')", "pseudoshuffle", "cardarea.lua:573 (game.lua:2383)",
       "initial deck shuffle at run start", via="CardArea:shuffle()",
       pool="G.deck.cards = 52 Card objects created in suit..rank string order (game.lua:2367), Fisher-Yates from the end"),
    _k("shuffle_new_round", "nr{ante}", "'nr'..G.GAME.round_resets.ante", "pseudoshuffle", "functions/state_events.lua:344",
       "deck shuffle when a blind starts", via="CardArea:shuffle", ante=True, pool="deck cards re-sorted by sort_id first"),
    _k("shuffle_cashout", "cashout{ante}", "'cashout'..G.GAME.round_resets.ante", "pseudoshuffle", "functions/button_callbacks.lua:2918",
       "deck shuffle on cash out", via="CardArea:shuffle", ante=True, pool="deck cards re-sorted by sort_id first"),
    _k("erratic", "erratic", "'erratic'", "pseudorandom_element", "game.lua:2342",
       "Erratic Deck: 52 independent draws of a random P_CARDS key for the starting deck",
       via="Game:start_run", pool="G.P_CARDS key-string order; results then sorted by suit..rank before card creation"),
    _k("idol", "idol{ante}", "'idol'..G.GAME.round_resets.ante", "pseudorandom_element", "functions/common_events.lua:2281",
       "The Idol target card (run start and after each boss)", via="reset_idol_card", ante=True,
       pool="non-stone cards of G.playing_cards -> sort_id (creation) order"),
    _k("mail", "mail{ante}", "'mail'..G.GAME.round_resets.ante", "pseudorandom_element", "functions/common_events.lua:2297",
       "Mail-In Rebate rank", via="reset_mail_rank", ante=True, pool="non-stone G.playing_cards -> sort_id order"),
    _k("ancient", "anc{ante}", "'anc'..G.GAME.round_resets.ante", "pseudorandom_element", "functions/common_events.lua:2308",
       "Ancient Joker suit", via="reset_ancient_card", ante=True,
       pool="{'Spades','Hearts','Clubs','Diamonds'} minus current suit, array order"),
    _k("castle", "cas{ante}", "'cas'..G.GAME.round_resets.ante", "pseudorandom_element", "functions/common_events.lua:2321",
       "Castle suit", via="reset_castle_card", ante=True, pool="non-stone G.playing_cards -> sort_id order"),
    _k("orbital", "orbital", "'orbital'", "pseudorandom_element", "functions/UI_definitions.lua:1515",
       "Orbital Tag hand offered at a blind (per ante per blind type)", via="blind select UI",
       pool="visible G.GAME.hands keys via pairs() -> LuaJIT HASH ORDER (not reproducible from source alone)",
       note="NOT EXTRACTED file."),
    _k("flipped_card", "flipped_card", "'flipped_card'", "pseudorandom", "cardarea.lua:602 / functions/common_events.lua:404",
       "challenge modifier flipped_cards (1/N face down)", note="Challenges only."),

    # ------------------------------------------------------------------ boss blinds
    _k("hook", "hook", "'hook'", "pseudorandom_element", "blind.lua:475", "The Hook: 2 random hand cards discarded after each hand",
       via="Blind:press_play", pool="copy of G.hand.cards -> sort_id order; 2 consecutive draws, picked card removed between them"),
    _k("cerulean_bell", "cerulean_bell", "'cerulean_bell'", "pseudorandom_element", "blind.lua:583", "Cerulean Bell forced card",
       via="Blind:drawn_to_hand", pool="G.hand.cards -> sort_id order"),
    _k("crimson_heart", "crimson_heart", "'crimson_heart'", "pseudorandom_element", "blind.lua:594", "Crimson Heart debuffed joker each hand",
       via="Blind:drawn_to_hand", pool="non-debuffed jokers (all if < 2) -> sort_id order"),
    _k("wheel_blind", "wheel", "'wheel'", "pseudorandom", "blind.lua:608", "The Wheel: card drawn face down if < 1/7",
       via="Blind:stay_flipped"),
    _k("amber_acorn", "aajk", "'aajk'", "pseudoshuffle", "blind.lua:197-201", "Amber Acorn: jokers flipped and shuffled (3 shuffles in a row)",
       via="G.jokers:shuffle('aajk')", pool="G.jokers.cards re-sorted by sort_id before each shuffle"),

    # ------------------------------------------------------------------ consumable effects
    _k("to_do", "to_do | false_to_do", "(area.config.type == 'title') and 'false_to_do' or 'to_do'", "pseudorandom_element",
       "card.lua:320 (set_ability), card.lua:2980 (end of round)", "To Do List target hand",
       pool="visible G.GAME.hands keys via pairs() -> LuaJIT HASH ORDER; re-rolls while equal to previous hand",
       note="'false_to_do' only for main-menu title cards."),
    _k("sigil", "sigil", "'sigil'", "pseudorandom_element", "card.lua:1233", "Sigil suit", pool="{'S','H','D','C'}"),
    _k("ouija", "ouija", "'ouija'", "pseudorandom_element", "card.lua:1247", "Ouija rank", pool="{'2'..'9','T','J','Q','K','A'} (13)"),
    _k("random_destroy", "random_destroy", "'random_destroy'", "pseudorandom_element", "card.lua:1293",
       "Familiar/Grim/Incantation: hand card destroyed", pool="G.hand.cards -> sort_id order"),
    _k("familiar_create", "familiar_create", "'familiar_create'", "pseudorandom_element", "card.lua:1320-1321",
       "Familiar: per created card, rank from {'J','Q','K'} THEN suit from {'S','H','D','C'} (2 draws, same stream)"),
    _k("grim_create", "grim_create", "'grim_create'", "pseudorandom_element", "card.lua:1324",
       "Grim: per created card, suit from {'S','H','D','C'} (rank fixed 'A')"),
    _k("incantation_create", "incantation_create", "'incantation_create'", "pseudorandom_element", "card.lua:1326-1327",
       "Incantation: per created card, rank from {'2'..'9','T'} THEN suit (2 draws)"),
    _k("spe_card", "spe_card", "'spe_card'", "pseudorandom_element", "card.lua:1336",
       "enhancement of each Familiar/Grim/Incantation card", pool="pools.ENHANCEMENTS minus m_stone (7), pool order"),
    _k("immolate", "immolate", "'immolate'", "pseudoshuffle", "card.lua:1344", "Immolate: first 5 of shuffled hand destroyed",
       pool="hand cards; pseudoshuffle re-sorts by sort_id before Fisher-Yates (the playing_card pre-sort is overridden)"),
    _k("ankh_choice", "ankh_choice", "'ankh_choice'", "pseudorandom_element", "card.lua:1434", "Ankh: joker kept/copied",
       pool="G.jokers.cards -> sort_id order"),
    _k("wheel_of_fortune", "wheel_of_fortune", "'wheel_of_fortune'", "pseudorandom + pseudorandom_element + pseudorandom",
       "card.lua:1470,1473,1484", "Wheel of Fortune: (1) success if < 1/4; (2) which editionless joker; (3) poll_edition guaranteed no-negative",
       pool="eligible_strength_jokers (editionless jokers) -> sort_id order", note="Three consecutive draws from ONE stream."),
    _k("ectoplasm", "ectoplasm", "'ectoplasm'", "pseudorandom_element", "card.lua:1473", "Ectoplasm: joker made Negative",
       pool="eligible_editionless_jokers -> sort_id order"),
    _k("hex", "hex", "'hex'", "pseudorandom_element", "card.lua:1473", "Hex: joker made Polychrome (others destroyed)",
       pool="eligible_editionless_jokers -> sort_id order"),
    _k("aura", "aura", "'aura'", "pseudorandom", "card.lua:1195", "Aura: edition on the selected card, poll_edition(guaranteed, no_neg)", via="poll_edition"),

    # ------------------------------------------------------------------ joker triggers
    _k("lucky_mult", "lucky_mult", "'lucky_mult'", "pseudorandom", "card.lua:988", "Lucky Card +20 Mult if < 1/5"),
    _k("lucky_money", "lucky_money", "'lucky_money'", "pseudorandom", "card.lua:1076", "Lucky Card $20 if < 1/15"),
    _k("glass", "glass", "'glass'", "pseudorandom", "functions/state_events.lua:961", "Glass Card shatters if < 1/4 (per scored glass card)"),
    _k("hallucination", "halu{ante}", "'halu'..G.GAME.round_resets.ante", "pseudorandom", "card.lua:2337",
       "Hallucination: tarot on pack open if < 1/2", ante=True),
    _k("invisible", "invisible", "'invisible'", "pseudorandom_element", "card.lua:2383", "Invisible Joker: which joker is copied",
       pool="other jokers -> sort_id order"),
    _k("perkeo", "perkeo", "'perkeo'", "pseudorandom_element", "card.lua:2417", "Perkeo: which consumable is copied",
       pool="G.consumeables.cards -> sort_id order"),
    _k("certificate_front", "cert_fr", "'cert_fr'", "pseudorandom_element", "card.lua:2467", "Certificate: created card",
       pool="G.P_CARDS key-string order"),
    _k("certificate_seal", "certsl", "'certsl'", "pseudorandom", "card.lua:2469", "Certificate seal (>0.75 Red, >0.5 Blue, >0.25 Gold, else Purple)"),
    _k("madness", "madness", "'madness'", "pseudorandom_element", "card.lua:2509", "Madness: joker destroyed",
       pool="other non-eternal, not-being-sliced jokers -> sort_id order"),
    _k("marble_front", "marb_fr", "'marb_fr'", "pseudorandom_element", "card.lua:2583", "Marble Joker stone card front",
       pool="G.P_CARDS key-string order"),
    _k("gros_michel", "gros_michel", "'gros_michel'", "pseudorandom", "card.lua:3020", "Gros Michel goes extinct at round end if < 1/6"),
    _k("cavendish", "cavendish", "'cavendish'", "pseudorandom", "card.lua:3020", "Cavendish destroyed at round end if < 1/1000"),
    _k("8ball", "8ball", "'8ball'", "pseudorandom", "card.lua:3107", "8 Ball: tarot per scored 8 if < 1/4"),
    _k("business", "business", "'business'", "pseudorandom", "card.lua:3177", "Business Card: $2 per scored face if < 1/2"),
    _k("bloodstone", "bloodstone", "'bloodstone'", "pseudorandom", "card.lua:3249", "Bloodstone: x1.5 per scored Heart if < 1/2"),
    _k("parking", "parking", "'parking'", "pseudorandom", "card.lua:3304", "Reserved Parking: $1 per held face if < 1/2"),
    _k("space", "space", "'space'", "pseudorandom", "card.lua:3420", "Space Joker: level up if < 1/4"),
    _k("misprint", "misprint", "pseudorandom('misprint', 0, 23)", "pseudorandom_int", "card.lua:3701", "Misprint +Mult = math.random(0, 23)"),

    # ------------------------------------------------------------------ defined but unused in vanilla
    _k("edition_deck", "edition_deck", "'edition_deck'", "pseudorandom_element", "back.lua:222",
       "deck config.edition hook: random playing card gets an edition", pool="G.playing_cards -> sort_id order",
       note="No vanilla deck sets config.edition (challenge/mod hook)."),
    _k("pack_generic", "pack_generic{ante}", "'pack_generic'..ante", "pseudorandom", "functions/common_events.lua:1953",
       "get_pack default key", ante=True, note="Only 'shop_pack' is ever passed."),
    _k("edition_generic", "edition_generic", "'edition_generic'", "pseudorandom", "functions/common_events.lua:2057",
       "poll_edition default key", note="Every vanilla caller passes a key."),
    _k("seed_special", "seed", "pseudoseed('seed') -> math.random()  (unseeded)", "pseudorandom", "functions/misc_functions.lua:298",
       "guard: the literal key 'seed' bypasses the keyed stream", note="Never called with 'seed' in vanilla."),
]

KEY_BY_NAME = {k["name"]: k for k in KEYS}

# Unseeded math.random() calls that touch gameplay objects (no pseudoseed; they advance the
# global LuaJIT state, which every keyed call re-seeds anyway, so they never affect parity):
UNSEEDED_GAMEPLAY = [
    ("functions/common_events.lua:1947", "first shop pack: p_buffoon_normal_<math.random(1,2)> (variants are content-identical)"),
    ("tag.lua:211", "Charm Tag: p_arcana_mega_<math.random(1,2)>"),
    ("tag.lua:226", "Meteor Tag: p_celestial_mega_<math.random(1,2)>"),
    ("card.lua:959", "Stone Card get_id() = -math.random(100, 1000000) (hand sort only; draws sort by sort_id, unaffected)"),
    ("functions/button_callbacks.lua:3071, game.lua:1457, UI_definitions.lua:3879/4233/6008", "pseudorandom_element(G.P_CARDS) with no seed: menu/splash/deck-preview art only"),
]

# Run-start keyed call order (Game:start_run, game.lua:2175-2389), for the oracle:
RUN_START_SEQUENCE = [
    "boss",                  # get_new_boss()                       game.lua:2177
    "Voucher{ante}",         # get_next_voucher_key() (+resamples)  game.lua:2178
    "Tag{ante}", "Tag{ante}",  # Small tag, Big tag                  game.lua:2179-2180
    "erratic*52 (Erratic Deck only)",                              # game.lua:2342
    "shuffle",               # self.deck:shuffle()                  game.lua:2383
    "idol{ante}", "mail{ante}", "anc{ante}", "cas{ante}",           # game.lua:2385-2389
]

# After EVERY blind (state_events.lua:236-280): idol, mail, anc, cas are re-rolled with the
# current ante.  When the blind was the Boss, ease_ante(1) is queued FIRST (state_events.lua:248),
# so the sequence is:  [ante += 1] -> Voucher{ante} -> idol{ante}, mail{ante}, anc{ante}, cas{ante}
# -> (cash out, button_callbacks.lua:2949-2953) Tag{ante} x2 -> reset_blinds -> boss.
# Small/Big blinds: just idol, mail, anc, cas.
ROUND_END_SEQUENCE = {
    "Small": ["idol{ante}", "mail{ante}", "anc{ante}", "cas{ante}"],
    "Big":   ["idol{ante}", "mail{ante}", "anc{ante}", "cas{ante}"],
    "Boss":  ["<ante += 1>", "Voucher{ante}", "idol{ante}", "mail{ante}", "anc{ante}", "cas{ante}",
              "Tag{ante}", "Tag{ante}", "boss"],
}


# ---------------------------------------------------------------------------------------
# Multiplayer mod (Phase 2 W2, NOTES_ORDER.md): how the keys above change under The Order
# (RunState.key_scope == "run") and Major League Balatro (RunState.ruleset == "mlb").
# Builders live in generate.Keys (gen_ante / ante_suffix / boss / order_* / VOUCHER_ORDER /
# new_round_shuffle(ante, state) / cashout_shuffle(ante, state) / halu_for); this is the inventory.
# ---------------------------------------------------------------------------------------

# MLB (The Order off, as the ruleset forces): ONLY the voucher draws change.
MLB_KEYS = {
    "voucher_pool":         "Voucher0   (culled base/upgrade pairs; redraw re-steps the same stream; 'Voucher0'..it after 1000)",
    "voucher_pool_fromtag": "Voucher0   (same stream as the shop voucher)",
}

# The Order: the RNG seed becomes '*'..seed for EVERY key (mod prefixes it before hashed_seed);
# then these constructions change.  Keys not listed keep their vanilla construction with the
# real ante (Tag{ante}, idol/mail/anc/cas{ante}, illusion, omen_globe, erratic, all joker
# probability keys except halu, ...).
ORDER_KEYS = {
    "boss":               "boss{ante}",
    "voucher_pool":       "Voucher0  (as MLB)",
    "voucher_pool_fromtag": "Voucher0",
    "shop_type":          "cdt0",
    "shop_pack":          "shop_pack0",
    "hallucination":      "halu0",
    "std_set":            "stdset0", "std_edition": "standard_edition0",
    "std_seal":           "stdseal0", "std_seal_type": "stdsealtype0",
    # inside create_card the ante reads 0 and key_append is rewritten:
    "rarity_roll":        "rarity0  (every joker source; Judgement with eternals enabled rolls 'order_jud_rarity' instead; The Soul rolls nothing)",
    "joker_pool":         "Joker{rarity}0 | Joker{rarity}0_sticker  (the mod's joker loop; legendary stays Joker4)",
    "edition_shop":       "ediJoker{rarity}0 | ediJoker{rarity}0_sticker  (one step per loop iteration; ediJoker4 for The Soul)",
    "eternal_perishable": "_etperJoker{rarity}0  (shop + pack, every iteration)",
    "rental":             "_rentJoker{rarity}0   (rentals enabled only)",
    "tarot_pool":         "TarotTarot0 | TarotTarot_pack0 (G.pack_cards)",
    "planet_pool":        "PlanetPlanet0 | PlanetPlanet_pack0",
    "spectral_pool":      "SpectralSpectral0 | SpectralSpectral_pack0",
    "soul_roll":          "soul_{type}0",
    "front":              "front{append}0   (append kept for Base/Enhanced: frontsho0, frontsta0)",
    "enhanced_pool":      "Enhanced{append}0",
    "resample":           "{base}  (the pool stream itself, re-stepped; {base}_resample{it} only after it > 1000; tags keep vanilla)",
    # shuffles / picks: keys and algorithm
    "shuffle_new_round":  "nr{ante}{blind_key}{blind_type}   + value-ranking shuffle (give_shufflevals)",
    "shuffle_cashout":    "cashout{ante}{blind_key}{blind_type}  (defeated blind) + value-ranking shuffle",
    "shuffle_start":      "shuffle  (key unchanged; value-ranking shuffle)",
    "immolate":           "immolate (key unchanged; value-ranking shuffle)",
    "idol":               "idol{ante}  (key unchanged; the mod's scored weighted walk)",
    "mail":               "mail{ante}  (key unchanged; count-weighted walk in rank order)",
    "castle":             "cas{ante}   (key unchanged; value-ranked pick)",
    "joker/card picks":   "hex, ankh_choice, ectoplasm, wheel_of_fortune, hook, cerulean_bell, crimson_heart, random_destroy, invisible, madness: keys unchanged; value-ranked pick",
    "to_do / orbital":    "keys unchanged; candidates in G.GAME.hands[k].order (= HANDLIST) instead of pairs() order",
}
