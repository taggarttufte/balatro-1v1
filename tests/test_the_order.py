"""The Order / Major League Balatro generation oracle: ``rng.generate`` with
``RunState.key_scope == "run"`` (The Order) and ``RunState.ruleset == "mlb"`` against the
game's own Lua WITH the BalatroMultiplayer mod's patches applied, executing in LuaJIT.

How the Lua side is built (nothing from the mod is copied into this repo):

* the vanilla reference (``_reference/balatro_src``) is loaded exactly as
  ``test_generate_oracle.LuaGenOracle`` does, but its text is first run through the mod's
  ``lovely/TheOrder.toml`` pattern patches (read from the installed mod at test time and
  applied with lovely's line-match + indent rule).  The two shuffle sites and the two
  seed lines live in other vanilla files; the harness repeats those vanilla lines verbatim
  and the same toml patches are applied to them;
* ``compatibility/TheOrder.lua`` (the mod's Lua: ``create_card`` wrapper, ``get_culled`` +
  the voucher overrides, ``MP.ante_based`` / ``order_round_based`` / ``sorted_hand_list``,
  the ``reset_idol_card`` / ``reset_mail_rank`` replacements, the Standard-pack card
  definition, the ``pseudoshuffle`` / ``pseudorandom_element`` overrides) is executed
  VERBATIM on top, with ``MP.should_use_the_order`` / ``MP.is_major_league_ruleset``
  (core.lua:104-115) and the few Steamodded entry points it touches stubbed -- see
  ``_LUA_MOD_STUBS`` for exactly what is stubbed and what each stub reproduces.

Then both sides run the same script (run start, antes 1-8, three shops per ante with
rerolls, every pack opened, creation paths, Voucher Tag, round picks, the nr/cashout
shuffles, joker picks) in three modes and every value is compared:

* ``order``   -- The Order on (``key_scope="run"``),
* ``mlb``     -- The Order off, MLB ruleset (``ruleset="mlb"``): vouchers only,
* ``vanilla`` -- The Order off, vanilla ruleset, mod still loaded: must equal plain vanilla
  (guards the stubs and the mod's unconditional Standard-pack override).

Skips cleanly when lupa, the extracted game Lua or the installed mod
(``BALATRO_MP_MOD_DIR``, default ``%APPDATA%/Balatro/Mods/Multiplayer``) is unavailable.
"""

from __future__ import annotations

import os
import random
import sys
import tomllib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MP_ROOT = HERE.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from rng import generate as G  # noqa: E402
from rng import pools as P  # noqa: E402
from tests import test_generate_oracle as base  # noqa: E402

MOD_DIR = Path(os.environ.get("BALATRO_MP_MOD_DIR",
                              os.path.join(os.environ.get("APPDATA", ""), "Balatro", "Mods", "Multiplayer")))
MOD_TOML = MOD_DIR / "lovely" / "TheOrder.toml"
MOD_LUA = MOD_DIR / "compatibility" / "TheOrder.lua"


def _mod_available() -> bool:
    return MOD_TOML.is_file() and MOD_LUA.is_file()


pytestmark = pytest.mark.skipif(
    not (base._lupa_available() and base._sources_available() and _mod_available()),
    reason="needs lupa (LuaJIT 2.1), _reference/balatro_src and the installed Multiplayer mod "
           "(set BALATRO_MP_MOD_DIR)",
)


# --------------------------------------------------------------------------------------
# lovely pattern patches, applied to source text in Python
# --------------------------------------------------------------------------------------

def _apply_pattern(text: str, pattern: str, position: str, payload: str, times=None) -> tuple:
    """One ``[patches.pattern]`` with ``match_indent = true``: every line whose stripped text
    equals the stripped pattern (multi-line patterns: consecutive lines) is replaced
    (``at``) or the payload is inserted ``before`` / ``after`` it, each payload line
    re-indented to the matched line's indent.  Returns ``(new_text, matches)``."""
    plines = [l.strip() for l in pattern.strip("\n").splitlines() if l.strip()]
    lines = text.splitlines()
    out, i, n = [], 0, 0
    while i < len(lines):
        block = lines[i:i + len(plines)]
        if plines and len(block) == len(plines) and all(b.strip() == p for b, p in zip(block, plines)) \
                and (times is None or n < times):
            indent = block[0][:len(block[0]) - len(block[0].lstrip())]
            pay = [indent + l if l.strip() else l for l in payload.strip("\n").splitlines()]
            if position == "at":
                out.extend(pay)
            elif position == "before":
                out.extend(pay)
                out.extend(block)
            elif position == "after":
                out.extend(block)
                out.extend(pay)
            else:  # pragma: no cover
                raise ValueError(position)
            i += len(plines)
            n += 1
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out), n


def _load_toml_patches() -> list:
    doc = tomllib.loads(MOD_TOML.read_text(encoding="utf-8"))
    return [p["pattern"] for p in doc["patches"] if "pattern" in p]


def _patch_text(text: str, targets, patches: list, applied: dict) -> str:
    """Apply every toml pattern patch whose ``target`` is in ``targets`` to ``text``; count
    matches per pattern in ``applied``."""
    for p in patches:
        if p["target"] not in targets:
            continue
        text, n = _apply_pattern(text, p["pattern"], p["position"], p["payload"], p.get("times"))
        applied[p["pattern"].strip()] = applied.get(p["pattern"].strip(), 0) + n
    return text


# Steamodded's get_current_pool (rarity.toml) resolves `_legendary` BEFORE the rarity roll, so
# The Soul does not step 'rarity..' -- the shared 'rarity0' stream under The Order.  One-line
# port of that Steamodded patch onto the vanilla line (generate.get_current_pool mirrors it).
_VANILLA_RARITY_LINE = "local rarity = _rarity or pseudorandom('rarity'..G.GAME.round_resets.ante..(_append or ''))"
_SMODS_RARITY_LINE = "local rarity = (_legendary and 4) or _rarity or pseudorandom('rarity'..G.GAME.round_resets.ante..(_append or ''))"

# --------------------------------------------------------------------------------------
# stubs for what the mod's Lua touches outside the vanilla files (our own Lua, documented)
# --------------------------------------------------------------------------------------

_LUA_MOD_STUBS = r"""
MP = {}
MP.ORDER, MP.MLB = false, false
function MP.should_use_the_order() return MP.ORDER end        -- core.lua:104-111 (lobby flag)
function MP.is_major_league_ruleset() return MP.MLB end      -- core.lua:113-115
function sendDebugMessage() end
SMODS = {}
-- SMODS.Booster:take_ownership_by_kind: the mod registers its Standard-pack card recipe; we keep it.
SMODS.Booster = {}
function SMODS.Booster:take_ownership_by_kind(kind, obj, silent) STD_BOOSTER = obj end
-- SMODS.poll_seal({mod = 10}) for vanilla seals == card.lua:1763-1772 (weights 4 x W, base 2%:
-- gate 1 - 4W*mod/(200W) = 1 - 0.02*mod; type thresholds 0.75/0.5/0.25 in pool order Red,Blue,Gold,Purple)
function SMODS.poll_seal(args)
  local ante = G.GAME.round_resets.ante
  local seal_poll = pseudorandom(pseudoseed('stdseal'..ante))
  if seal_poll > 1 - 0.02*(args.mod or 1) then
    local seal_type = pseudorandom(pseudoseed('stdsealtype'..ante))
    if seal_type > 0.75 then return 'Red' elseif seal_type > 0.5 then return 'Blue'
    elseif seal_type > 0.25 then return 'Gold' else return 'Purple' end
  end
end
-- SMODS.create_card(t): the argument table -> vanilla create_card, then set_edition/set_seal
function SMODS.create_card(t)
  local c = create_card(t.set, t.area, t.legendary, t.rarity, t.skip_materialize, t.soulable, t.key, t.key_append)
  if t.edition then c:set_edition(t.edition) end
  if t.seal then c:set_seal(t.seal) end
  return c
end
function SMODS.size_of_pool(pool)
  local size = 0
  for _, v in pairs(pool) do if v ~= 'UNAVAILABLE' then size = size + 1 end end
  return size
end
-- SMODS.get_next_vouchers (vanilla path; utils.lua): == get_next_voucher_key with a spawn set
function SMODS.get_next_vouchers(vouchers)
  vouchers = vouchers or {spawn = {}}
  local _pool, _pool_key = get_current_pool('Voucher')
  for i = #vouchers + 1, math.min(SMODS.size_of_pool(_pool), G.GAME.starting_params.vouchers_in_shop + (G.GAME.modifiers.extra_vouchers or 0)) do
    local center = pseudorandom_element(_pool, pseudoseed(_pool_key))
    local it = 1
    while center == 'UNAVAILABLE' or vouchers.spawn[center] do
      it = it + 1
      center = pseudorandom_element(_pool, pseudoseed(_pool_key..'_resample'..it))
    end
    vouchers[#vouchers + 1] = center
    vouchers.spawn[center] = true
  end
  return vouchers
end
-- SMODS.Rank / SMODS.Suit registration order (game_object.lua) and the rank fields the mod reads
SMODS.Rank = {obj_buffer = {'2','3','4','5','6','7','8','9','10','Jack','Queen','King','Ace'}}
SMODS.Suit = {obj_buffer = {'Diamonds','Clubs','Hearts','Spades'}}
SMODS.Ranks = {}
for i, k in ipairs(SMODS.Rank.obj_buffer) do SMODS.Ranks[k] = {nominal = (i <= 9) and (i + 1) or ((i == 13) and 11 or 10), face = (i >= 10 and i <= 12) or nil} end
SMODS.Stickers = {eternal = {should_apply = false}, perishable = {should_apply = false}, rental = {should_apply = false}}
"""

# Harness additions.  The three functions that repeat vanilla lines from files the oracle does
# not load (game.lua:2167-2168, state_events.lua:344, button_callbacks.lua:2918) are written
# with those lines verbatim so the toml patches can be applied to them.
_LUA_ORDER_HARNESS_RAW = r"""
function SETUP_RUN_PATCHED(seed)
  G.GAME = init_game_object()
  G.GAME.pseudorandom.seed = seed
  local self = G
  for k, v in pairs(self.GAME.pseudorandom) do if v == 0 then self.GAME.pseudorandom[k] = pseudohash(k..self.GAME.pseudorandom.seed) end end
  self.GAME.pseudorandom.hashed_seed = pseudohash(self.GAME.pseudorandom.seed)
  G.GAME.selected_back = {pos = {x=0, y=0}}
  G.GAME.tags = {}
  G.GAME.starting_params.vouchers_in_shop = 1
  G.GAME.blind = {config = {blind = {}}}
  G.GAME.blind_on_deck = nil
  G.jokers = new_area('joker'); G.consumeables = new_area('joker')
  G.shop_jokers = nil; G.shop_vouchers = nil; G.shop_booster = nil
  G.pack_cards = new_area('shop')
  G.playing_cards = {}
  G.OVERLAY_MENU = nil
end
function SET_BLIND(key, kind) G.GAME.blind = {config = {blind = {key = key}}}; G.GAME.blind_on_deck = kind end
function SHUFFLE_NR()
  G.deck:shuffle('nr'..G.GAME.round_resets.ante)
end
function SHUFFLE_CASHOUT()
  G.deck:shuffle('cashout'..G.GAME.round_resets.ante)
end
"""

_LUA_ORDER_HARNESS = r"""
function SET_ORDER(on) MP.ORDER = on end
function SET_MLB(on) MP.MLB = on end
function EFFECTIVE_SEED() return G.GAME.pseudorandom.seed end
function SHOP_VOUCHER()
  local t = SMODS.get_next_vouchers()
  G.GAME.current_round.voucher = t[1]
  return t[1]
end
-- the Steamodded Card:open branch for Standard packs: booster_obj:create_card(self, i) -> SMODS.create_card
function PACK_LOOP_SMODS(self)
  local pack_cards = {}
  local _size = self.ability.extra
  for i = 1, _size do
    local card
    if self.ability.name:find('Standard') then
      card = SMODS.create_card(STD_BOOSTER.create_card(STD_BOOSTER, self, i))
    else
      card = PACK_LOOP_ONE(self, i)
    end
    pack_cards[i] = card
  end
  return pack_cards
end
function OPEN_PACK_SMODS(key)
  local pack = {ability = {name = G.P_CENTERS[key].name, extra = G.P_CENTERS[key].config.extra}, T = {x=0,y=0}}
  local cards = PACK_LOOP_SMODS(pack)
  for _, c in ipairs(cards) do G.pack_cards:emplace(c) end
  return cards
end
-- playing-card tables with everything the mod's give_shufflevals / reset_* read:
-- csv items 'H_7' or 'H_7:m_glass:Red:foil' (center key, seal, edition type; '' for none)
local SUITS = {S = 'Spades', H = 'Hearts', C = 'Clubs', D = 'Diamonds'}
local RANKS = {['2']={'2',2},['3']={'3',3},['4']={'4',4},['5']={'5',5},['6']={'6',6},['7']={'7',7},['8']={'8',8},
               ['9']={'9',9},T={'10',10},J={'Jack',11},Q={'Queen',12},K={'King',13},A={'Ace',14}}
local EFFECT = {c_base='Base', m_bonus='Bonus Card', m_mult='Mult Card', m_wild='Wild Card', m_glass='Glass Card',
                m_steel='Steel Card', m_stone='Stone Card', m_gold='Gold Card', m_lucky='Lucky Card'}
function SET_PLAYING_CARDS_EX(csv)
  G.playing_cards = {}
  local i = 0
  for item in string.gmatch(csv, '[^,]+') do
    i = i + 1
    local key, center, seal, ed = string.match(item, '^([^:]+):?([^:]*):?([^:]*):?([^:]*)$')
    center = (center ~= '' and center) or 'c_base'
    local r = RANKS[string.sub(key, 3, 3)]
    local c = {sort_id = i, key = key,
               ability = {effect = EFFECT[center], set = (center == 'c_base') and 'Default' or 'Enhanced'},
               base = {value = r[1], id = r[2], suit = SUITS[string.sub(key, 1, 1)]},
               config = {center = {key = center}, center_key = center}}
    if seal ~= '' then c.seal = seal end
    if ed ~= '' then c.edition = {type = ed, [ed] = true} end
    G.playing_cards[i] = c
  end
end
function DECK_FROM_PLAYING_CARDS()
  G.deck = {cards = {}}
  for i, c in ipairs(G.playing_cards) do G.deck.cards[i] = c end
  function G.deck:shuffle(_seed) pseudoshuffle(self.cards, pseudoseed(_seed or 'shuffle')) end
end
function DECK_ORDER()
  local out = {}
  for i, c in ipairs(G.deck.cards) do out[i] = c.key end
  return table.concat(out, ',')
end
-- deck order by card KIND (key + center + seal + edition): the mod's shuffle ranks the cards by
-- value and cards of identical kind tie; which physical copy lands where then depends on the
-- order of the list the shuffle was given (G.deck.cards in the game, creation order in the
-- engine), so identity among identical cards is not part of the contract.
function CARD_KIND(c)
  local k = (c.config.center.key == 'm_stone') and 'Stone' or c.key   -- stone cards: rank/suit are dead
  return k .. ':' .. c.config.center.key .. ':' .. (c.seal or '') .. ':' .. (c.edition and c.edition.type or '')
end
function DECK_KINDS()
  local out = {}
  for i, c in ipairs(G.deck.cards) do out[i] = CARD_KIND(c) end
  return table.concat(out, ',')
end
function IMMOLATE_KINDS()
  local temp = {}
  for i, c in ipairs(G.playing_cards) do temp[i] = c end
  pseudoshuffle(temp, pseudoseed('immolate'))
  local out = {}
  for i = 1, 5 do out[i] = CARD_KIND(temp[i]) end
  return table.concat(out, ',')
end
function SHUFFLE_KEY(key) G.deck:shuffle(key) end
function RESETS_ORDER()
  G.GAME.current_round.ancient_card.suit = nil
  reset_idol_card(); reset_mail_rank(); reset_ancient_card(); reset_castle_card()
  local cr = G.GAME.current_round
  return table.concat({cr.idol_card.rank .. '|' .. cr.idol_card.suit, cr.mail_card.rank, cr.ancient_card.suit,
                       cr.castle_card.suit}, ';')
end
function CASTLE_RANK_SUIT()
  -- castle stores only the suit; re-run the same pick to expose the card (same stream step count either way)
  return G.GAME.current_round.castle_card.suit
end
function PICK_JOKERS(csv, key)
  local list, i = {}, 0
  for k in string.gmatch(csv, '[^,]+') do i = i + 1; list[i] = {sort_id = i, ability = {set = 'Joker'}, config = {center = {key = k}}} end
  local _, idx = pseudorandom_element(list, pseudoseed(key))
  return idx
end
function PICK_CARDS(key)
  local _, idx = pseudorandom_element(G.playing_cards, pseudoseed(key))
  return idx
end
function IMMOLATE_FIRST5()
  local temp = {}
  for i, c in ipairs(G.playing_cards) do temp[i] = c end
  pseudoshuffle(temp, pseudoseed('immolate'))
  local out = {}
  for i = 1, 5 do out[i] = temp[i].key end
  return table.concat(out, ',')
end
function ORBITAL_ORDER()
  local _poker_hands = MP.sorted_hand_list()
  local h = pseudorandom_element(_poker_hands, pseudoseed('orbital'))
  return h
end
function TO_DO_ORDER()
  local _poker_hands = MP.sorted_hand_list(nil)
  local h = pseudorandom_element(_poker_hands, pseudoseed('to_do'))
  return h
end
function HALU_KEY() return 'halu'..MP.ante_based() end
"""


class LuaOrderOracle:
    """Vanilla generation Lua + the Multiplayer mod's TheOrder patches and Lua, in LuaJIT."""

    EXPECTED_PATCHES = {   # toml pattern (stripped) -> minimum matches against our loaded text
        "local _, boss = pseudorandom_element(eligible_bosses, pseudoseed('boss'))": 1,
        "for k, v in pairs(self.GAME.pseudorandom) do if v == 0 then self.GAME.pseudorandom[k] = pseudohash(k..self.GAME.pseudorandom.seed) end end": 1,
        "local poll = pseudorandom(pseudoseed((_key or 'pack_generic')..G.GAME.round_resets.ante))*cume": 1,
        "G.deck:shuffle('nr'..G.GAME.round_resets.ante)": 1,
        "G.deck:shuffle('cashout'..G.GAME.round_resets.ante)": 1,
        "local polled_rate = pseudorandom(pseudoseed('cdt'..G.GAME.round_resets.ante))*total_rate": 1,
        "center = pseudorandom_element(_pool, pseudoseed(_pool_key..'_resample'..it))": 2,   # create_card + get_next_voucher_key
        "if forced_key and not G.GAME.banned_keys[forced_key] then": 1,
        "if (area == G.shop_jokers) or (area == G.pack_cards) then": 1,
    }

    def __init__(self):
        from lupa import luajit21 as lj

        patches = _load_toml_patches()
        self.applied: dict = {}
        common = base.COMMON.read_text(encoding="utf-8", errors="replace")
        assert common.count(_VANILLA_RARITY_LINE) == 1, "common_events.lua rarity line drifted"
        common = common.replace(_VANILLA_RARITY_LINE, _SMODS_RARITY_LINE)
        # the toml's get_pack patch targets Steamodded's copy of the line; vanilla has the same line
        common = _patch_text(common, {"functions/common_events.lua", '=[SMODS _ "src/overrides.lua"]'}, patches, self.applied)
        ccfs = _patch_text(base._create_card_for_shop(), {"functions/UI_definitions.lua"}, patches, self.applied)
        harness_raw = _patch_text(_LUA_ORDER_HARNESS_RAW,
                                  {"game.lua", "functions/state_events.lua", "functions/button_callbacks.lua"},
                                  patches, self.applied)
        for pat, n in self.EXPECTED_PATCHES.items():
            assert self.applied.get(pat, 0) >= n, "toml patch did not apply (%d): %s" % (self.applied.get(pat, 0), pat)

        self.L = lj.LuaRuntime(unpack_returned_tuples=True)
        ex = self.L.execute
        ex(base._LUA_PRE)
        ex(base.MISC.read_text(encoding="utf-8", errors="replace"))
        ex(common)
        ex(base._LUA_STUBS)
        ex("function INIT_PROTOS(self) " + base._body_init_item_prototypes() + "\nend")
        ex("G.save_progress = function() end; INIT_PROTOS(G)")
        ex("function init_game_object() " + base._body_init_game_object() + "\nend")
        ex(ccfs)
        # one pack card of the vanilla loop (used for every non-Standard pack)
        ex("function PACK_LOOP_ONE(self, i)\n local pack_cards = {}\n local _size = 1\n"
           + base._pack_loop().split("\n", 2)[2].replace("for i = 1, _size do", "do", 1)
           + "\n return pack_cards[i]\nend")
        ex(base._LUA_HARNESS)
        ex("for k, v in pairs(G.P_CARDS) do v.key = k end")
        ex("FULL_PROFILE()")
        ex(_LUA_MOD_STUBS)
        ex(MOD_LUA.read_text(encoding="utf-8", errors="replace"))     # the mod's Lua, verbatim
        ex(harness_raw)
        ex(_LUA_ORDER_HARNESS)
        self.g = self.L.globals()

    def __getattr__(self, name):
        return getattr(self.g, name)


_ORACLE = None


def oracle() -> LuaOrderOracle:
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = LuaOrderOracle()
    return _ORACLE


# --------------------------------------------------------------------------------------
# scripts
# --------------------------------------------------------------------------------------

def _state(seed: str, mode: str, stake: int = 1) -> G.RunState:
    st = G.RunState.for_stake(seed, stake)
    if mode == "order":
        st.key_scope = G.KEY_SCOPE_RUN
    elif mode == "mlb":
        st.ruleset = G.RULESET_MLB
    return st


def _setup(L, seed: str, mode: str, stake: int):
    L.SET_ORDER(mode == "order")
    L.SET_MLB(mode == "mlb")
    L.SETUP_RUN_PATCHED(seed)
    L.SET_STAKE_MODS(stake)


def _card_keys(n=52) -> list:
    return G.build_starting_deck(G.RunState("X"))[:n]


def run_script(seed: str, mode: str, *, antes: int = 8, stake: int = 1, buy_in_shop=None,
               showman_from_ante=None, banned=()):
    """Shops / rerolls / packs / bosses / tags / vouchers / Voucher Tag / shuffles / round picks."""
    L = oracle()
    mism = []

    def check(label, lua_v, py_v):
        if lua_v != py_v:
            mism.append("%s %s [%s]: lua=%r py=%r" % (mode, seed, label, lua_v, py_v))

    _setup(L, seed, mode, stake)
    for k in banned:
        L.BAN(k)
    st = _state(seed, mode, stake)
    st.banned_keys = set(banned)
    check("effective_seed", L.EFFECTIVE_SEED(), st.rng.seed)

    deck = G.build_starting_deck(st)
    L.SET_PLAYING_CARDS_EX(",".join(deck))
    L.DECK_FROM_PLAYING_CARDS()
    l_boss = L.get_new_boss(); L.G.GAME.round_resets.blind_choices.Boss = l_boss
    l_vou = L.SHOP_VOUCHER()
    l_t1 = L.get_next_tag_key(); l_t2 = L.get_next_tag_key()
    L.SHUFFLE_KEY(None)   # CardArea:shuffle() at run start -> 'shuffle'
    l_resets = L.RESETS_ORDER()
    rs = G.start_run(st)
    check("boss1", l_boss, rs.boss); check("voucher1", l_vou, rs.voucher)
    check("tagS1", l_t1, rs.tag_small); check("tagB1", l_t2, rs.tag_big)
    check("shuffle", L.DECK_ORDER(), ",".join(rs.deck))
    picks = _py_resets(st, deck, rs)
    check("resets1", l_resets, picks)

    for ante in range(1, antes + 1):
        if ante > 1:
            L.SET_ANTE(ante)
            l_vou = L.SHOP_VOUCHER()
            l_t1 = L.get_next_tag_key(); l_t2 = L.get_next_tag_key()
            l_boss = L.get_new_boss()
            info = G.defeat_boss(st)
            check("voucher%d" % ante, l_vou, info["voucher"])
            check("tagS%d" % ante, l_t1, info["tag_small"]); check("tagB%d" % ante, l_t2, info["tag_big"])
            check("boss%d" % ante, l_boss, info["boss"])
        if showman_from_ante and ante >= showman_from_ante and not st.showman:
            L.OWN("j_ring_master"); st.acquire("j_ring_master")
        for shop_i, (bkey, btype) in enumerate((("bl_small", "Small"), ("bl_big", "Big"), (l_boss, "Boss"))):
            # blind start: 'nr' shuffle with the blind on G.GAME.blind / blind_on_deck
            L.SET_BLIND(bkey, btype); st.blind_key, st.blind_type = bkey, btype
            L.NEW_ROUND(); st.new_round()
            L.SHUFFLE_NR()
            check("a%d %s nr" % (ante, btype), L.DECK_ORDER(),
                  ",".join(G.shuffle_deck(st, deck, G.Keys.new_round_shuffle(st.ante))))
            # cash out: 'cashout' shuffle with the DEFEATED blind still set
            L.SHUFFLE_CASHOUT()
            check("a%d %s cashout" % (ante, btype), L.DECK_ORDER(),
                  ",".join(G.shuffle_deck(st, deck, G.Keys.cashout_shuffle(st.ante))))
            L.OPEN_SHOP()
            shop = G.generate_shop(st)
            check("a%d s%d shelf" % (ante, shop_i), L.SHELF(), base.py_cards(shop.cards))
            l_packs = L.PACKS()
            check("a%d s%d packs" % (ante, shop_i), base._norm_first_pack(l_packs), base._norm_first_pack(base.py_packs(shop)))
            for r in range(2):
                L.REROLL(); G.reroll_shop(st, shop)
                check("a%d s%d reroll%d" % (ante, shop_i, r), L.SHELF(), base.py_cards(shop.cards))
            if buy_in_shop is not None and shop_i == buy_in_shop and shop.cards:
                L.BUY_SLOT(1); st.acquire(shop.cards[0]); shop.cards.pop(0)
            for pi, pk in enumerate(l_packs.split(";")):
                l_pc = L.OPEN_PACK_SMODS(pk)
                p_pc = G.open_pack(st, pk)
                check("a%d s%d pack%d %s" % (ante, shop_i, pi, pk), L.PACK_DESC(l_pc), base.py_cards(p_pc))
                L.DISCARD_PACK(); st.release_pack(p_pc)
            if shop_i == 1:   # a Voucher Tag draw once per ante (mid-shop, vouchers on display)
                check("a%d voucher_tag" % ante, L.get_next_voucher_key(True), G.next_voucher(st, from_tag=True))
            L.CLOSE_SHOP(); st.release_shop(shop)
            l_resets = L.RESETS_ORDER()
            check("a%d %s resets" % (ante, btype), l_resets, _py_resets(st, deck, None))
        check("a%d used_jokers" % ante, L.USED_JOKERS(),
              ",".join(sorted(k for k in st.used_jokers if k[:2] in ("j_", "c_"))))
        check("a%d bosses_used" % ante, L.BOSSES_USED(),
              ",".join("%s=%d" % (k, st.bosses_used[k]) for k in sorted(st.bosses_used)))
    return mism


def _py_resets(st, deck, rs) -> str:
    """Same serialisation as the Lua RESETS_ORDER (idol rank|suit; mail rank; ancient; castle suit)."""
    if rs is not None:
        # start_run already drew them; recover from the RunStart
        idol = rs.idol
        idol_s = G._RANK_VALUE[G._RANK_ID[idol[2]]] + "|" + G._SUIT_NAME[idol[0]]
        mail_s = G._RANK_VALUE[G._RANK_ID[rs.mail[2]]]
        castle_s = G._SUIT_NAME[rs.castle[0]]
        return ";".join([idol_s, mail_s, rs.ancient_suit, castle_s])
    picks = G.reset_round_picks(st, deck)
    return ";".join([picks["idol"][0] + "|" + picks["idol"][1], picks["mail"], picks["ancient"], picks["castle"][1]])


def run_creation_paths(seed: str, mode: str, stake: int = 1):
    """Consumable / joker / tag creation paths, Aura, joker picks, immolate, to_do/orbital."""
    L = oracle()
    mism = []

    def check(label, lua_v, py_v):
        if lua_v != py_v:
            mism.append("%s %s [%s]: lua=%r py=%r" % (mode, seed, label, lua_v, py_v))

    _setup(L, seed, mode, stake)
    st = _state(seed, mode, stake)
    L.SET_ANTE(3); st.ante = 3
    cases = [
        ("judgement", "CC_DESC('Joker', G.jokers, false, nil, nil, nil, nil, 'jud')", "judgement"),
        ("soul", "CC_DESC('Joker', G.jokers, true, nil, nil, nil, nil, 'sou')", "soul"),
        ("wraith", "CC_DESC('Joker', G.jokers, nil, 0.99, nil, nil, nil, 'wra')", "wraith"),
        ("riff1", "CC_DESC('Joker', G.jokers, nil, 0, nil, nil, nil, 'rif')", "riff_raff"),
        ("riff2", "CC_DESC('Joker', G.jokers, nil, 0, nil, nil, nil, 'rif')", "riff_raff"),
        ("top_up", "CC_DESC('Joker', G.jokers, nil, 0, nil, nil, nil, 'top')", "top_up_tag"),
        ("emperor1", "CC_DESC('Tarot', G.consumeables, nil, nil, nil, nil, nil, 'emp')", "emperor"),
        ("emperor2", "CC_DESC('Tarot', G.consumeables, nil, nil, nil, nil, nil, 'emp')", "emperor"),
        ("priestess", "CC_DESC('Planet', G.consumeables, nil, nil, nil, nil, nil, 'pri')", "high_priestess"),
        ("sixth", "CC_DESC('Spectral', G.consumeables, nil, nil, nil, nil, nil, 'sixth')", "sixth_sense"),
        ("8ball", "CC_DESC('Tarot', G.consumeables, nil, nil, nil, nil, nil, '8ba')", "8_ball"),
        ("purple", "CC_DESC('Tarot', G.consumeables, nil, nil, nil, nil, nil, '8ba')", "purple_seal"),
        ("judgement2", "CC_DESC('Joker', G.jokers, false, nil, nil, nil, nil, 'jud')", "judgement"),
        ("soul2", "CC_DESC('Joker', G.jokers, true, nil, nil, nil, nil, 'sou')", "soul"),
        ("hallucination", "CC_DESC('Tarot', G.consumeables, nil, nil, nil, nil, nil, 'hal')", "hallucination"),
    ]
    for label, lua_src, spec in cases:
        check(label, L.L.eval(lua_src), base.py_card(G.create_from_spec(st, spec)))
    L.NEW_ROUND(); st.new_round()
    L.OPEN_SHOP()
    shop = G.generate_shop(st)
    check("shelf", L.SHELF(), base.py_cards(shop.cards))
    check("rare_tag", L.L.eval("CC_DESC('Joker', G.shop_jokers, nil, 1, nil, nil, nil, 'rta')"),
          base.py_card(G.create_from_spec(st, "rare_tag")))
    check("uncommon_tag", L.L.eval("CC_DESC('Joker', G.shop_jokers, nil, 0.9, nil, nil, nil, 'uta')"),
          base.py_card(G.create_from_spec(st, "uncommon_tag")))
    check("voucher_tag1", L.get_next_voucher_key(True), G.next_voucher(st, from_tag=True))
    check("voucher_tag2", L.get_next_voucher_key(True), G.next_voucher(st, from_tag=True))
    for i in range(3):
        check("aura%d" % i, L.L.eval("EDITION_NAME(poll_edition('aura', nil, true, true))"), G.aura(st) or "")
    check("halu_key", L.HALU_KEY(), G.Keys.halu_for(st))
    # joker picks (hex / ankh / ectoplasm / wheel) over a joker list with duplicates
    jokers = ["j_joker", "j_greedy_joker", "j_joker", "j_blueprint", "j_greedy_joker", "j_joker"]
    for key, fn in (("hex", G.hex_), ("ankh_choice", G.ankh), ("ectoplasm", G.ectoplasm)):
        for rep in range(2):
            check("%s%d" % (key, rep), L.PICK_JOKERS(",".join(jokers), key), fn(st, jokers) + 1)
    # modified deck: enhancements / seals / editions / stone + duplicates -> stdval ordering
    rnd = random.Random(seed)
    deck = _card_keys()
    centers = ["", "m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky"]
    seals = ["", "Red", "Blue", "Gold", "Purple"]
    eds = ["", "foil", "holo", "polychrome"]   # no Negative: unreachable on playing cards in vanilla (and 0 in the mod's stdval)
    items = []
    for k in deck + [rnd.choice(deck) for _ in range(8)]:   # duplicates
        items.append((k, rnd.choice(centers) if rnd.random() < 0.4 else "",
                      rnd.choice(seals) if rnd.random() < 0.25 else "",
                      rnd.choice(eds) if rnd.random() < 0.25 else ""))
    L.SET_PLAYING_CARDS_EX(",".join(":".join(it) for it in items))
    L.DECK_FROM_PLAYING_CARDS()
    py_cards = [{"suit": k[0], "rank": k[2], "center": c or "c_base", "seal": s or None, "edition": e or None}
                for k, c, s, e in items]
    for rep in range(3):
        L.SHUFFLE_KEY("shuffle")
        py_order = ",".join(_kind_of(c) for c in G.shuffle_deck(st, py_cards, "shuffle"))
        check("mod_deck_shuffle%d" % rep, L.DECK_KINDS(), py_order)
    L.SET_BLIND("bl_hook", "Boss"); st.blind_key, st.blind_type = "bl_hook", "Boss"
    L.SHUFFLE_NR()
    check("mod_deck_nr", L.DECK_KINDS(), ",".join(_kind_of(c) for c in G.shuffle_deck(st, py_cards, G.Keys.new_round_shuffle(st.ante))))
    check("mod_deck_immolate", L.IMMOLATE_KINDS(), ",".join(_kind_of(c) for c in G.immolate(st, py_cards)))
    for key in ("cas3", "hook", "cerulean_bell", "random_destroy"):
        # compared by kind: G.playing_cards keeps creation order on both sides here, but ties
        # among identical cards are still resolved by the (ported) sort -- the kind must match
        _, idx = (G.order_pick(st, py_cards, key) if mode == "order" else st.rng.pseudorandom_element(py_cards, key))
        check("pick_cards_" + key, _kind_of(py_cards[L.PICK_CARDS(key) - 1]), _kind_of(py_cards[idx]))
    check("resets_mod_deck", L.RESETS_ORDER(), _py_resets(st, py_cards, None))
    if mode == "order":
        visible = {h for h in P.HANDLIST if h not in ("Flush Five", "Flush House", "Five of a Kind")}
        for rep in range(3):
            check("orbital%d" % rep, L.ORBITAL_ORDER(), G.orbital_hand(st, visible))
            check("to_do%d" % rep, L.TO_DO_ORDER(), G.to_do_hand(st, visible))
    return mism


def _key_of(c) -> str:
    return c if isinstance(c, str) else c["suit"] + "_" + c["rank"]


def _kind_of(c) -> str:
    if isinstance(c, str):
        return c + ":c_base::"
    k = "Stone" if c["center"] == "m_stone" else _key_of(c)
    return "%s:%s:%s:%s" % (k, c["center"], c["seal"] or "", c["edition"] or "")


# --------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------

def _seeds(n: int, salt: int) -> list:
    rnd = random.Random(salt)
    return ["".join(rnd.choice(base.SEED_CORPUS) for _ in range(8)) for _ in range(n)]


SEEDS = ["EXAMPLE1", "ALEEB", "7LB2WVPK", "IMMOLATE"] + _seeds(18, 20260821)
MODES = ["order", "mlb", "vanilla"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", SEEDS)
def test_full_run_parity(seed, mode):
    mism = run_script(seed, mode)
    assert not mism, "\n".join(mism[:40])


@pytest.mark.parametrize("scenario", ["buy", "gold_stake", "showman_buy", "banned"])
@pytest.mark.parametrize("seed", SEEDS[:8])
def test_order_scenarios(seed, scenario):
    kw = {"buy": dict(buy_in_shop=0), "gold_stake": dict(stake=8), "showman_buy": dict(showman_from_ante=2, buy_in_shop=1),
          "banned": dict(banned=("bl_hook", "bl_wall", "j_joker", "c_fool", "p_buffoon_normal_1",
                                 "j_mr_bones", "j_luchador", "j_matador", "j_chicot",
                                 "v_hieroglyph", "v_petroglyph", "v_directors_cut", "v_retcon", "tag_boss"))}[scenario]
    mism = run_script(seed, "order", antes=4, **kw)
    assert not mism, "\n".join(mism[:40])


@pytest.mark.parametrize("seed", SEEDS[:8])
def test_mlb_banned_attrition(seed):
    """MLB with the Attrition bans: vouchers from the culled 'Voucher0' pool honour bans."""
    mism = run_script(seed, "mlb", antes=6,
                      banned=("j_mr_bones", "j_luchador", "j_matador", "j_chicot",
                              "v_hieroglyph", "v_petroglyph", "v_directors_cut", "v_retcon", "tag_boss",
                              "bl_wall", "bl_final_vessel"))
    assert not mism, "\n".join(mism[:40])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", SEEDS[:12])
def test_creation_paths(seed, mode):
    mism = run_creation_paths(seed, mode)
    assert not mism, "\n".join(mism[:40])


@pytest.mark.parametrize("seed", SEEDS[:6])
def test_creation_paths_gold_stake_order(seed):
    """Gold stake under The Order: the '_etper'/'_rent' sticker polls and the '_sticker' joker queue."""
    mism = run_creation_paths(seed, "order", stake=8)
    assert not mism, "\n".join(mism[:40])


def test_toml_patches_applied():
    """Every Order key site the toml patches must have matched our loaded vanilla text."""
    L = oracle()
    for pat, n in LuaOrderOracle.EXPECTED_PATCHES.items():
        assert L.applied.get(pat, 0) >= n, pat


def test_mlb_differs_from_vanilla_only_in_vouchers():
    """ruleset='mlb' (The Order off): identical streams everywhere except the voucher draws."""
    for seed in SEEDS[:10]:
        a, b = G.RunState(seed), G.RunState(seed, ruleset=G.RULESET_MLB)
        ra, rb = G.start_run(a), G.start_run(b)
        assert (ra.boss, ra.tag_small, ra.tag_big, ra.deck) == (rb.boss, rb.tag_small, rb.tag_big, rb.deck)
        assert a.rng.seed == b.rng.seed == seed
        for ante in range(1, 5):
            if ante > 1:
                ia, ib = G.defeat_boss(a), G.defeat_boss(b)
                assert (ia["boss"], ia["tag_small"], ia["tag_big"]) == (ib["boss"], ib["tag_small"], ib["tag_big"])
            for _ in range(3):
                a.new_round(); b.new_round()
                sa, sb = G.generate_shop(a), G.generate_shop(b)
                assert base.py_cards(sa.cards) == base.py_cards(sb.cards)
                assert sa.boosters == sb.boosters
                for pk in sa.boosters:
                    if pk:
                        pa, pb = G.open_pack(a, pk), G.open_pack(b, pk)
                        assert base.py_cards(pa) == base.py_cards(pb)
                        a.release_pack(pa); b.release_pack(pb)
                a.release_shop(sa); b.release_shop(sb)
        # the MLB voucher stream is the run-global 'Voucher0' one
        assert "Voucher0" in list(b.rng.keys()) and "Voucher1" not in list(b.rng.keys())
        assert "Voucher1" in list(a.rng.keys())


def test_key_scope_reseeds_and_guards():
    st = G.RunState("ABC")
    assert st.rng.seed == "ABC"
    st.key_scope = G.KEY_SCOPE_RUN
    assert st.rng.seed == "*ABC" and st.effective_seed == "*ABC"
    st.key_scope = G.KEY_SCOPE_ANTE
    assert st.rng.seed == "ABC"
    G.start_run(st)
    with pytest.raises(RuntimeError):
        st.key_scope = G.KEY_SCOPE_RUN
    # clone keeps the scope and the stream
    st2 = G.RunState("ABC", key_scope=G.KEY_SCOPE_RUN)
    G.start_run(st2)
    c = st2.clone()
    assert c.key_scope == "run" and c.rng.seed == "*ABC"
    assert G.next_tag(c) == G.next_tag(st2)
