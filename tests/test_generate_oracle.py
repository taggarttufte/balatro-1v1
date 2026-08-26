"""Generation-layer oracle: ``rng.generate`` vs Balatro's own Lua.

The game's REAL generation functions -- ``get_current_pool``, ``create_card``, ``poll_edition``,
``get_pack``, ``get_next_voucher_key``, ``get_next_tag_key``, ``get_new_boss``,
``reset_idol_card``/``reset_mail_rank``/``reset_ancient_card``/``reset_castle_card``,
``create_card_for_shop`` (UI_definitions.lua:742-800), the ``Card:open`` pack loop
(card.lua:1726-1781) and the primitives from misc_functions.lua -- are loaded VERBATIM from
``_reference/balatro_src`` into LuaJIT 2.1 (lupa) at test time, together with the game's own
``Game:init_item_prototypes`` and ``Game:init_game_object``.  Only ``G``/``Card``/UI plumbing is
stubbed (see ``_LUA_HARNESS``; the ``Card`` stub reproduces what ``Card:set_ability`` and
``Card:remove`` do to ``G.GAME.used_jokers``, card.lua:349-354 / 4741-4747).

Both sides are then driven through identical scripts (run start; antes 1-3; three shops per
ante with rerolls; every pack opened; purchases / Showman / bans / Gold-stake stickers /
fresh-profile locks) and every generated key, front, edition, seal, sticker, voucher, tag,
boss, ``used_jokers`` and ``bosses_used`` is compared.

Conventions (per NOTES_CORE.md): ``jit.off()`` in the runtime; no FFI pointer punning; no
doubles cross the Python/Lua boundary at all -- only key strings, ints and booleans (the few
numeric ``_rarity`` arguments are Lua literals parsed by LuaJIT itself, exactly as in the game
source).  No game Lua text lives in this file; every slice is located by line number and the
boundary lines are asserted so source drift fails loudly.

Skips cleanly when ``lupa`` or the extracted game Lua is unavailable.  The ``pairs(G.GAME.hands)``
order test additionally needs the game's own ``lua51.dll`` (LuaJIT 2.0.5; fixed string hash) and
is skipped without it -- set ``BALATRO_DIR`` if Balatro is not at the default Steam path.
"""

from __future__ import annotations

import ctypes
import os
import random
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
MP_ROOT = HERE.parent
if str(MP_ROOT) not in sys.path:
    sys.path.insert(0, str(MP_ROOT))

from rng import generate as G  # noqa: E402
from rng import pools as P  # noqa: E402

SRC = MP_ROOT / "_reference" / "balatro_src"
MISC = SRC / "functions" / "misc_functions.lua"
COMMON = SRC / "functions" / "common_events.lua"
UIDEF = SRC / "functions" / "UI_definitions.lua"
GAME = SRC / "game.lua"
CARD = SRC / "card.lua"
BALATRO_DIR = Path(os.environ.get("BALATRO_DIR", r"C:\Program Files (x86)\Steam\steamapps\common\Balatro"))
GAME_LUA_DLL = BALATRO_DIR / "lua51.dll"

SEED_CORPUS = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"


def _lupa_available() -> bool:
    try:
        from lupa import luajit21  # noqa: F401
    except Exception:
        return False
    return True


def _sources_available() -> bool:
    return all(p.is_file() for p in (MISC, COMMON, UIDEF, GAME, CARD))


pytestmark = pytest.mark.skipif(
    not (_lupa_available() and _sources_available()),
    reason="needs lupa (LuaJIT 2.1) and the extracted game Lua under _reference/balatro_src",
)


# --------------------------------------------------------------------------------------
# verbatim slices of the game source, located by line with boundary assertions
# --------------------------------------------------------------------------------------

def _slice(path: Path, first: int, last: int, starts: str, ends: str) -> str:
    """1-based inclusive line slice; asserts the first/last line prefixes so that a changed
    reference file cannot silently shift what gets loaded."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    a, b = lines[first - 1], lines[last - 1]
    assert a.strip().startswith(starts), "%s:%d drifted: %r" % (path.name, first, a)
    assert b.strip().startswith(ends), "%s:%d drifted: %r" % (path.name, last, b)
    return "\n".join(lines[first - 1:last])


def _body_init_item_prototypes() -> str:
    return _slice(GAME, 217, 842, "--Initialize all prototypes", "end")


def _body_init_game_object() -> str:
    return _slice(GAME, 1863, 2015, "local bosses_used = {}", "}")


def _create_card_for_shop() -> str:
    return _slice(UIDEF, 742, 800, "function create_card_for_shop(area)", "end")


def _pack_loop() -> str:
    return _slice(CARD, 1726, 1781, "local _size = self.ability.extra", "end")


def _hands_constructor() -> str:
    return _slice(GAME, 2001, 2014, "hands = {", "}")


# --------------------------------------------------------------------------------------
# the stubs (our own Lua, not game text)
# --------------------------------------------------------------------------------------

_LUA_PRE = r"""
jit.off()
_RELEASE_MODE = true
love = {filesystem = {getInfo=function() return true end, createDirectory=function() end, append=function() end}}
G = {SETTINGS = {profile = 1, tutorial_complete = true, GAMESPEED = 1}, C = {WHITE = {1,1,1,1}}}
Event = function(t) return t end
G.E_MANAGER = {add_event = function(self, e) end}
G.ARGS = {}
G.CARD_W, G.CARD_H = 1, 1
G.sort_id = 0
"""

# re-asserted AFTER the game files load (misc_functions.lua defines its own localize etc.)
_LUA_STUBS = r"""
function localize(args) if type(args)=='table' then return '<loc:'..tostring(args.key or args.type)..'>' end return '<loc:'..tostring(args)..'>' end
function HEX(h) return h end
function delay() end
function play_sound() end
function ease_dollars() end
function discover_card() end
function inc_career_stat() end
function check_for_unlock() end
function create_shop_card_ui() end
function convert_save_to_meta() end
function get_compressed() return 'return {}' end
function STR_UNPACK(s) return {} end
"""

_LUA_HARNESS = r"""
function new_area(kind)
  local a = {cards = {}, config = {card_limit = 99, type = kind}, T = {x=0,y=0,w=0,h=0}}
  function a:emplace(card) card.area = self; table.insert(self.cards, card) end
  function a:remove_card(card)
    for i = #self.cards, 1, -1 do if self.cards[i] == card then table.remove(self.cards, i) end end
    card.area = nil
    return card
  end
  return a
end

-- Minimal Card: what Card:init / Card:set_ability / Card:remove do that generation can observe.
function Card(X, Y, W, H, card, center, params)
  G.sort_id = G.sort_id + 1
  local self = {T = {x=X, y=Y}, sort_id = G.sort_id, params = params or {}, children = {}, states = {visible = true},
                config = {card = card, center = center, center_key = center.key}}
  self.ability = {name = center.name, set = center.set, rarity = center.rarity,
                  consumeable = center.consumeable and center.config or nil,
                  extra = center.config and center.config.extra or nil}
  if not G.OVERLAY_MENU then                                         -- card.lua:349-354
    for k, v in pairs(G.P_CENTERS) do if v.name == self.ability.name then G.GAME.used_jokers[k] = true end end
  end
  local _ = math.random(); _ = math.random(); _ = math.random()       -- card.lua:46-50 discard_pos
  function self:set_edition(e) self.edition = e end
  function self:set_eternal(b) self.ability.eternal = b end
  function self:set_perishable(b) self.ability.perishable = b end
  function self:set_rental(b) self.ability.rental = b end
  function self:set_seal(s) self.seal = s end
  function self:start_materialize() end
  function self:set_cost() end
  function self:juice_up() end
  function self:add_to_deck() end
  function self:remove()                                              -- card.lua:4741-4747
    if self.area then self.area:remove_card(self) end
    if not G.OVERLAY_MENU then
      for k, v in pairs(G.P_CENTERS) do
        if v.name == self.ability.name then
          if not next(find_joker(self.ability.name, true)) then G.GAME.used_jokers[k] = nil end
        end
      end
    end
  end
  return self
end

function SETUP_RUN(seed)
  G.GAME = init_game_object()
  G.GAME.pseudorandom.seed = seed
  G.GAME.pseudorandom.hashed_seed = pseudohash(seed)
  G.GAME.selected_back = {pos = {x=0, y=0}}
  G.GAME.tags = {}
  G.jokers = new_area('joker'); G.consumeables = new_area('joker')
  G.shop_jokers = nil; G.shop_vouchers = nil; G.shop_booster = nil
  G.pack_cards = new_area('shop')
  G.playing_cards = {}
  G.OVERLAY_MENU = nil
end
function BAN(key) G.GAME.banned_keys[key] = true end
function SET_ANTE(a) G.GAME.round_resets.ante = a end
function SET_STAKE_MODS(stake)
  if stake >= 4 then G.GAME.modifiers.enable_eternals_in_shop = true end
  if stake >= 7 then G.GAME.modifiers.enable_perishables_in_shop = true end
  if stake >= 8 then G.GAME.modifiers.enable_rentals_in_shop = true end
end
function FULL_PROFILE()
  for k, v in pairs(G.P_CENTERS) do v.unlocked = true; v.discovered = true end
  for k, v in pairs(G.P_BLINDS) do v.discovered = true end
  for k, v in pairs(G.P_TAGS) do v.discovered = true end
end

-- game.lua:3111-3160 (fresh shop) without the UI
function OPEN_SHOP()
  G.shop_jokers = new_area('shop'); G.shop_vouchers = new_area('shop'); G.shop_booster = new_area('shop')
  for i = 1, G.GAME.shop.joker_max - #G.shop_jokers.cards do
    G.shop_jokers:emplace(create_card_for_shop(G.shop_jokers))
  end
  if G.GAME.current_round.voucher then
    G.shop_vouchers:emplace(Card(0,0,1,1, nil, G.P_CENTERS[G.GAME.current_round.voucher], {}))
  end
  G.GAME.current_round.used_packs = G.GAME.current_round.used_packs or {}
  for i = 1, 2 do
    if not G.GAME.current_round.used_packs[i] then
      G.GAME.current_round.used_packs[i] = get_pack('shop_pack').key
    end
  end
end
function NEW_ROUND() G.GAME.current_round.used_packs = {} end        -- state_events.lua:301
-- button_callbacks.lua:2873-2883
function REROLL()
  for i = #G.shop_jokers.cards, 1, -1 do
    local c = G.shop_jokers:remove_card(G.shop_jokers.cards[i]); c:remove()
  end
  for i = 1, G.GAME.shop.joker_max - #G.shop_jokers.cards do
    G.shop_jokers:emplace(create_card_for_shop(G.shop_jokers))
  end
end
-- toggle_shop -> G.shop:remove() -> CardArea:remove -> Card:remove
function CLOSE_SHOP()
  for _, area in ipairs({G.shop_jokers, G.shop_vouchers, G.shop_booster}) do
    for i = #area.cards, 1, -1 do area.cards[i]:remove() end
    area.cards = nil
  end
end
function BUY_SLOT(i)
  local c = G.shop_jokers:remove_card(G.shop_jokers.cards[i])
  if c.ability.set == 'Joker' then G.jokers:emplace(c) else G.consumeables:emplace(c) end
end
function OWN(key) G.jokers:emplace(Card(0,0,1,1, nil, G.P_CENTERS[key], {})) end
function OPEN_PACK(key)
  local pack = {ability = {name = G.P_CENTERS[key].name, extra = G.P_CENTERS[key].config.extra}, T = {x=0,y=0}}
  local cards = PACK_LOOP(pack)
  for _, c in ipairs(cards) do G.pack_cards:emplace(c) end
  return cards
end
function DISCARD_PACK()
  for i = #G.pack_cards.cards, 1, -1 do G.pack_cards.cards[i]:remove() end
end

-- serialisation: everything crosses to Python as strings / ints / booleans
function CARD_DESC(c)
  local ed = c.edition and ((c.edition.negative and 'negative') or (c.edition.polychrome and 'polychrome')
             or (c.edition.holo and 'holo') or (c.edition.foil and 'foil')) or ''
  return table.concat({c.config.center.key, c.config.card and c.config.card.key or '', ed, c.seal or '',
                       c.ability.eternal and '1' or '0', c.ability.perishable and '1' or '0', c.ability.rental and '1' or '0'}, '|')
end
function SHELF() local t = {} for i, c in ipairs(G.shop_jokers.cards) do t[i] = CARD_DESC(c) end return table.concat(t, ';') end
function PACKS() return (G.GAME.current_round.used_packs[1] or '') .. ';' .. (G.GAME.current_round.used_packs[2] or '') end
function PACK_DESC(cards) local t = {} for i, c in ipairs(cards) do t[i] = CARD_DESC(c) end return table.concat(t, ';') end
function USED_JOKERS()
  local t = {}
  for k, v in pairs(G.GAME.used_jokers) do
    if v and (string.sub(k, 1, 2) == 'j_' or string.sub(k, 1, 2) == 'c_') then t[#t+1] = k end
  end
  table.sort(t)
  return table.concat(t, ',')
end
function BOSSES_USED()
  local keys = {}
  for k in pairs(G.GAME.bosses_used) do keys[#keys+1] = k end
  table.sort(keys)
  local t = {}
  for i, k in ipairs(keys) do t[i] = k .. '=' .. G.GAME.bosses_used[k] end
  return table.concat(t, ',')
end
function SHUFFLE_KEYS(csv, seedkey)
  local list, i = {}, 0
  for k in string.gmatch(csv, '[^,]+') do i = i + 1; list[i] = {sort_id = i, key = k} end
  pseudoshuffle(list, pseudoseed(seedkey))
  local out = {}
  for j, c in ipairs(list) do out[j] = c.key end
  return table.concat(out, ',')
end
function ERRATIC()
  local t = {}
  for k, v in pairs(G.P_CARDS) do local _, kk = pseudorandom_element(G.P_CARDS, pseudoseed('erratic')); t[#t+1] = kk end
  table.sort(t)
  return table.concat(t, ',')
end
function SET_PLAYING_CARDS(csv)
  G.playing_cards = {}
  local i = 0
  for k in string.gmatch(csv, '[^,]+') do
    i = i + 1
    G.playing_cards[i] = {sort_id = i, ability = {effect = 'Base'}, base = {value = string.sub(k, 3, 3), suit = string.sub(k, 1, 1), id = i}, key = k}
  end
end
function RESETS()
  G.GAME.current_round.ancient_card.suit = nil
  reset_idol_card(); reset_mail_rank(); reset_ancient_card(); reset_castle_card()
  local cr = G.GAME.current_round
  return table.concat({cr.idol_card.rank .. cr.idol_card.suit, cr.mail_card.rank, cr.ancient_card.suit, cr.castle_card.suit}, '|')
end
function CC_DESC(...) return CARD_DESC(create_card(...)) end
function EDITION_NAME(e)
  if not e then return '' end
  return (e.negative and 'negative') or (e.polychrome and 'polychrome') or (e.holo and 'holo') or (e.foil and 'foil') or ''
end
"""


class LuaGenOracle:
    """The game's generation Lua in LuaJIT 2.1 (JIT off) with a 'full' or 'fresh' profile."""

    def __init__(self, profile: str):
        from lupa import luajit21 as lj

        self.L = lj.LuaRuntime(unpack_returned_tuples=True)
        ex = self.L.execute
        ex(_LUA_PRE)
        ex(MISC.read_text(encoding="utf-8", errors="replace"))
        ex(COMMON.read_text(encoding="utf-8", errors="replace"))
        ex(_LUA_STUBS)
        ex("function INIT_PROTOS(self) " + _body_init_item_prototypes() + "\nend")
        ex("G.save_progress = function() end; INIT_PROTOS(G)")
        ex("function init_game_object() " + _body_init_game_object() + "\nend")
        ex(_create_card_for_shop())
        ex("function PACK_LOOP(self)\n local pack_cards = {}\n" + _pack_loop() + "\n return pack_cards\nend")
        ex(_LUA_HARNESS)
        ex("for k, v in pairs(G.P_CARDS) do v.key = k end")
        if profile == "full":
            ex("FULL_PROFILE()")
        self.version = self.L.eval("jit.version")
        self.g = self.L.globals()

    def __getattr__(self, name):
        return getattr(self.g, name)


_ORACLES: dict = {}


def oracle(profile: str) -> LuaGenOracle:
    if profile not in _ORACLES:
        _ORACLES[profile] = LuaGenOracle(profile)
    return _ORACLES[profile]


# --------------------------------------------------------------------------------------
# Python-side serialisation identical to CARD_DESC
# --------------------------------------------------------------------------------------

def py_card(c: G.CardGen) -> str:
    return "|".join([c.key, c.front or "", c.edition or "", c.seal or "",
                     "1" if c.eternal else "0", "1" if c.perishable else "0", "1" if c.rental else "0"])


def py_cards(cards) -> str:
    return ";".join(py_card(c) for c in cards)


def py_packs(shop: G.ShopContents) -> str:
    return ";".join(p or "" for p in shop.boosters)


def _norm_first_pack(s: str) -> str:
    # The forced first pack's art suffix is 'p_buffoon_normal_'..math.random(1,2) from the UNSEEDED
    # global state (common_events.lua:1945-1948): cosmetic, identical contents.
    a, _, b = s.partition(";")
    if a.startswith("p_buffoon_normal_"):
        a = "p_buffoon_normal_X"
    return a + ";" + b


# --------------------------------------------------------------------------------------
# the scripted run, executed on both sides in lock-step
# --------------------------------------------------------------------------------------

def run_script(seed: str, *, antes: int = 3, buy_in_shop=None, stake: int = 1, banned=(),
               showman_from_ante=None, profile: str = "full"):
    L = oracle(profile)
    mism = []

    def check(label, lua_v, py_v):
        if lua_v != py_v:
            mism.append("%s [%s]: lua=%r py=%r" % (seed, label, lua_v, py_v))

    L.SETUP_RUN(seed)
    for k in banned:
        L.BAN(k)
    L.SET_STAKE_MODS(stake)
    if profile == "full":
        st = G.RunState.for_stake(seed, stake)
    else:
        st = G.RunState.fresh_profile(seed, stake=stake)
        st.enable_eternals_in_shop, st.enable_perishables_in_shop, st.enable_rentals_in_shop = stake >= 4, stake >= 7, stake >= 8
    st.banned_keys = set(banned)

    # run start draws (game.lua:2177-2180), then the deck shuffle
    l_boss = L.get_new_boss(); L.G.GAME.round_resets.blind_choices.Boss = l_boss
    l_vou = L.get_next_voucher_key(); L.G.GAME.current_round.voucher = l_vou
    l_t1 = L.get_next_tag_key(); l_t2 = L.get_next_tag_key()
    rs = G.start_run(st)
    check("boss1", l_boss, rs.boss); check("voucher1", l_vou, rs.voucher)
    check("tagS1", l_t1, rs.tag_small); check("tagB1", l_t2, rs.tag_big)
    check("shuffle", L.SHUFFLE_KEYS(",".join(G.build_starting_deck(st)), "shuffle"), ",".join(rs.deck))

    for ante in range(1, antes + 1):
        if ante > 1:
            L.SET_ANTE(ante)
            l_vou = L.get_next_voucher_key(); L.G.GAME.current_round.voucher = l_vou
            l_t1 = L.get_next_tag_key(); l_t2 = L.get_next_tag_key()
            l_boss = L.get_new_boss()
            info = G.defeat_boss(st)
            check("voucher%d" % ante, l_vou, info["voucher"])
            check("tagS%d" % ante, l_t1, info["tag_small"]); check("tagB%d" % ante, l_t2, info["tag_big"])
            check("boss%d" % ante, l_boss, info["boss"])
        if showman_from_ante and ante >= showman_from_ante and not st.showman:
            L.OWN("j_ring_master")
            st.acquire("j_ring_master")
        for shop_i in range(3):
            L.NEW_ROUND(); st.new_round()
            L.OPEN_SHOP()
            shop = G.generate_shop(st)
            check("a%d s%d shelf" % (ante, shop_i), L.SHELF(), py_cards(shop.cards))
            l_packs = L.PACKS()
            check("a%d s%d packs" % (ante, shop_i), _norm_first_pack(l_packs), _norm_first_pack(py_packs(shop)))
            for r in range(2):
                L.REROLL(); G.reroll_shop(st, shop)
                check("a%d s%d reroll%d" % (ante, shop_i, r), L.SHELF(), py_cards(shop.cards))
            if buy_in_shop is not None and shop_i == buy_in_shop and shop.cards:
                L.BUY_SLOT(1); st.acquire(shop.cards[0]); shop.cards.pop(0)
            for pi, pk in enumerate(l_packs.split(";")):
                l_pc = L.OPEN_PACK(pk)
                p_pc = G.open_pack(st, pk)
                check("a%d s%d pack%d %s" % (ante, shop_i, pi, pk), L.PACK_DESC(l_pc), py_cards(p_pc))
                L.DISCARD_PACK(); st.release_pack(p_pc)
            L.CLOSE_SHOP(); st.release_shop(shop)
        check("a%d used_jokers" % ante, L.USED_JOKERS(),
              ",".join(sorted(k for k in st.used_jokers if k[:2] in ("j_", "c_"))))
        check("a%d bosses_used" % ante, L.BOSSES_USED(),
              ",".join("%s=%d" % (k, st.bosses_used[k]) for k in sorted(st.bosses_used)))
    return mism


def run_creation_paths(seed: str):
    """Consumable / joker / tag creation paths, Voucher Tag, Aura, Erratic, round resets."""
    L = oracle("full")
    mism = []

    def check(label, lua_v, py_v):
        if lua_v != py_v:
            mism.append("%s [%s]: lua=%r py=%r" % (seed, label, lua_v, py_v))

    L.SETUP_RUN(seed)
    st = G.RunState(seed)
    L.SET_ANTE(2); st.ante = 2
    # (label, Lua call with the game's literal arguments, generate.CREATE_SPECS name)
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
        ("soul2", "CC_DESC('Joker', G.jokers, true, nil, nil, nil, nil, 'sou')", "soul"),
    ]
    for label, lua_src, spec in cases:
        check(label, L.L.eval(lua_src), py_card(G.create_from_spec(st, spec)))
    L.NEW_ROUND(); st.new_round()
    L.OPEN_SHOP()
    shop = G.generate_shop(st)
    check("shelf", L.SHELF(), py_cards(shop.cards))
    check("rare_tag", L.L.eval("CC_DESC('Joker', G.shop_jokers, nil, 1, nil, nil, nil, 'rta')"),
          py_card(G.create_from_spec(st, "rare_tag")))
    check("uncommon_tag", L.L.eval("CC_DESC('Joker', G.shop_jokers, nil, 0.9, nil, nil, nil, 'uta')"),
          py_card(G.create_from_spec(st, "uncommon_tag")))
    check("voucher_tag1", L.get_next_voucher_key(True), G.next_voucher(st, from_tag=True))
    check("voucher_tag2", L.get_next_voucher_key(True), G.next_voucher(st, from_tag=True))
    for i in range(3):
        check("aura%d" % i, L.L.eval("EDITION_NAME(poll_edition('aura', nil, true, true))"), G.aura(st) or "")
    check("erratic", L.ERRATIC(), ",".join(sorted(G.build_starting_deck(st, erratic=True))))
    deck = G.build_starting_deck(st)
    L.SET_PLAYING_CARDS(",".join(deck))
    idol, _ = st.rng.pseudorandom_element(deck, G.Keys.idol(st.ante))
    mail, _ = st.rng.pseudorandom_element(deck, G.Keys.mail(st.ante))
    anc, _ = st.rng.pseudorandom_element(["Spades", "Hearts", "Clubs", "Diamonds"], G.Keys.ancient(st.ante))
    cas, _ = st.rng.pseudorandom_element(deck, G.Keys.castle(st.ante))
    check("resets", L.RESETS(), "|".join([idol[2] + idol[0], mail[2], anc, cas[0]]))
    return mism


# --------------------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------------------

def _seeds(n: int, salt: int) -> list:
    rnd = random.Random(salt)
    return ["".join(rnd.choice(SEED_CORPUS) for _ in range(8)) for _ in range(n)]


SCENARIOS = {
    "base": {},
    "buy": {"buy_in_shop": 0},
    "gold_stake": {"stake": 8},
    "banned": {"banned": ("bl_hook", "bl_wall", "j_joker", "c_fool", "p_buffoon_normal_1")},
    "showman_buy": {"showman_from_ante": 2, "buy_in_shop": 1},
    "fresh_profile": {"profile": "fresh"},
    "fresh_profile_buy": {"profile": "fresh", "buy_in_shop": 2},
}
SCRIPT_SEEDS = ["EXAMPLE1", "ALEEB", "7LB2WVPK"] + _seeds(12, 1234)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("seed", SCRIPT_SEEDS)
def test_shop_pack_voucher_boss_tag_parity(seed, scenario):
    mism = run_script(seed, **SCENARIOS[scenario])
    assert not mism, "\n".join(mism)


@pytest.mark.parametrize("seed", ["EXAMPLE1", "ALEEB"] + _seeds(18, 99))
def test_creation_paths_parity(seed):
    mism = run_creation_paths(seed)
    assert not mism, "\n".join(mism)


def test_oracle_is_luajit_with_jit_off():
    L = oracle("full")
    assert str(L.version).startswith("LuaJIT 2.1")
    assert L.L.eval("(jit.status())") is False  # parenthesised: first return value only


def test_keys_module_agreement():
    """generate.Keys and keys.py (Agent B) must build identical strings."""
    from rng import keys as K
    for ante in (0, 1, 3, 12):
        for app in ("", "sho", "buf", "jud"):
            assert G.Keys.rarity(ante, app) == K.rarity_key(ante, app)
            for r in (1, 2, 3):
                assert G.Keys.joker_pool(r, app, ante) == K.pool_key("Joker", ante, app, rarity=r)
            for t in ("Tarot", "Planet", "Spectral", "Voucher", "Tag", "Enhanced"):
                assert G.Keys.center_pool(t, app, ante) == K.pool_key(t, ante, app)
        assert G.Keys.joker_pool(4, "sou", ante, legendary=True) == K.pool_key("Joker", ante, "sou", rarity=4, legendary=True) == "Joker4"
    assert G.Keys.resample("Joker1sho1", 2) == K.resample_key("Joker1sho1", 2) == "Joker1sho1_resample2"
    assert set(G.CREATE_SPECS[k]["key_append"] for k in G.CREATE_SPECS) <= set(K.KEY_APPENDS)


def test_hands_pairs_order_is_a_permutation():
    assert sorted(G.HANDS_PAIRS_ORDER) == sorted(P.HANDLIST)
    assert len(set(G.HANDS_PAIRS_ORDER)) == 12


@pytest.mark.skipif(not GAME_LUA_DLL.is_file(), reason="needs the game's lua51.dll (set BALATRO_DIR)")
def test_hands_pairs_order_matches_game_dll():
    """``pairs(G.GAME.hands)`` order inside the game's OWN LuaJIT (2.0.5, fixed string hash).
    The verbatim ``hands = {...}`` constructor from Game:init_game_object is executed through
    the Lua C API via ctypes; the order must equal generate.HANDS_PAIRS_ORDER.  (lupa's LuaJIT
    2.1 randomises its string-hash seed per VM and cannot check this.)"""
    chunk = "local " + _hands_constructor() + """
local t = {}
for k, v in pairs(hands) do t[#t+1] = k end
return jit.version .. '|' .. table.concat(t, ',')
"""
    dll = ctypes.CDLL(str(GAME_LUA_DLL))
    dll.luaL_newstate.restype = ctypes.c_void_p
    dll.luaL_openlibs.argtypes = [ctypes.c_void_p]
    dll.luaL_loadstring.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    dll.lua_pcall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    dll.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
    dll.lua_tolstring.restype = ctypes.c_char_p
    dll.lua_close.argtypes = [ctypes.c_void_p]
    orders = set()
    versions = set()
    for _ in range(3):  # separate states: the order must not depend on VM start-up
        L = dll.luaL_newstate()
        try:
            dll.luaL_openlibs(L)
            assert dll.luaL_loadstring(L, chunk.encode()) == 0, dll.lua_tolstring(L, -1, None)
            assert dll.lua_pcall(L, 0, 1, 0) == 0, dll.lua_tolstring(L, -1, None)
            version, order = dll.lua_tolstring(L, -1, None).decode().split("|")
        finally:
            dll.lua_close(L)
        versions.add(version)
        orders.add(order)
    assert versions == {"LuaJIT 2.0.5"}, versions
    assert orders == {",".join(G.HANDS_PAIRS_ORDER)}, orders
