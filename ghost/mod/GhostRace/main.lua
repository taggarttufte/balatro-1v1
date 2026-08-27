-- GhostRace — live ghost racing against a local Python sidecar (balatro-1v1, G2).
--
-- Loads AFTER the Multiplayer mod (priority 10000001 > their 10000000) and wraps a
-- handful of MP.GHOST functions.  With a NORMAL replay everything delegates to the
-- originals — replay mode is untouched.  When the loaded replay carries a `_live`
-- marker (written by `python -m ghost.live`), this mod:
--   * sources the enemy's Nemesis hands from the sidecar's inbox file,
--   * owns PvP resolution with the SERVER rule (strict compare; exact tie takes
--     nobody's life) against the agent's FINAL score — no index-lag, ever,
--   * paces the score reveals on a wall-clock timer,
--   * appends the human's events (nemesis start, hands, results, blind fails) to the
--     outbox file the sidecar tails.
--
-- Contracts reproduced here (mod-source recon 2026-08-27, cited file:line in
-- ghost/G2_DESIGN.md): resolve_pvp_hands_exhausted returns "won"|"game_over"|"continue"
-- and never calls win_game() itself; resolve_pvp_mid_hand calls win_game() on the
-- terminal win; MP.GAME.end_pvp is the real round-end signal; MP.GAME.won must be set
-- before win_game(); enemy life loss is a silent decrement; own life loss decrements
-- MP.GAME.lives + MP.UI.ease_lives(-1) + the comeback/no-gold config side effects.
-- No Multiplayer code is copied; originals are saved and called through.

local json = require("json")

GR = {
    live = false,
    outbox = nil,
    inbox = nil,
    buffer = {},          -- [ante] = { hands = {...}, final = <number|string> }
    started = {},         -- [ante] = true once nemesis_start was emitted
    inbox_size = -1,
    inbox_seen = 0,
    pending_exhausted = nil,
    match_over = false,
    last_poll = 0,
    last_reveal = 0,
    POLL_S = 0.3,
    REVEAL_S = 2.5,
}

local ORIG = {
    load = MP.GHOST.load,
    clear = MP.GHOST.clear,
    get_enemy_hands = MP.GHOST.get_enemy_hands,
    advance_hand = MP.GHOST.advance_hand,
    resolve_pvp_mid_hand = MP.GHOST.resolve_pvp_mid_hand,
    resolve_pvp_hands_exhausted = MP.GHOST.resolve_pvp_hands_exhausted,
    resolve_round_fail = MP.GHOST.resolve_round_fail,
    start_advance_sequence = MP.GHOST._start_advance_sequence,
}

-------------------------------------------------------------------------------
-- small helpers
-------------------------------------------------------------------------------

local function numstr(x)
    -- Talisman makes G.GAME.chips a Big whose tostring comma-groups ("1,073");
    -- emit a plain machine-parsable decimal instead.
    local n = tonumber(x)
    if n then return string.format("%.0f", n) end
    return (tostring(x):gsub(",", ""))
end

local function current_ante()
    return G.GAME and G.GAME.round_resets and G.GAME.round_resets.ante or 0
end

function GR.emit(event, fields)
    if not GR.outbox then return end
    fields = fields or {}
    fields.e = event
    fields.ts = os.time()
    local ok, line = pcall(json.encode, fields)
    if ok then NFS.append(GR.outbox, line .. "\n") end
end

function GR.arm(replay)
    local live = replay._live
    GR.live = true
    GR.outbox, GR.inbox = live.outbox, live.inbox
    GR.buffer, GR.started = {}, {}
    GR.inbox_size, GR.inbox_seen = -1, 0
    GR.pending_exhausted, GR.match_over = nil, false
    GR.last_poll, GR.last_reveal = 0, 0
    if live.bootstrap and live.bootstrap.ante then
        GR.buffer[tonumber(live.bootstrap.ante)] = {
            hands = live.bootstrap.hands or {},
            final = live.bootstrap.final,
        }
    end
    GR.emit("session_start", { seed = replay.seed, deck = replay.deck,
                               stake = replay.stake })
    sendDebugMessage("GhostRace: live session armed (seed " ..
                     tostring(replay.seed) .. ")", "GHOSTRACE")
end

function GR.disarm(reason)
    if GR.live and not GR.match_over and reason then
        GR.emit("match_end", { winner = reason })
    end
    GR.live = false
    GR.outbox, GR.inbox = nil, nil
end

function GR.ensure_start(ante)
    if not GR.started[ante] then
        GR.started[ante] = true
        GR.emit("nemesis_start", { ante = ante, lives = MP.GAME.lives })
    end
end

function GR.emit_result(ante, chips, loser)
    local b = GR.buffer[ante]
    GR.emit("pvp_result", {
        ante = ante,
        human_score = numstr(chips),
        agent_score = b and numstr(b.final) or "?",
        loser = loser,                       -- "human" | "agent" | nil (tie)
        human_lives = MP.GAME.lives,         -- POST-resolution, per the protocol
    })
end

function GR.revealed_all()
    return #MP.GHOST._hands > 0 and MP.GHOST._hand_idx >= #MP.GHOST._hands
end

function GR.reveal_all()
    local guard = 0
    while #MP.GHOST._hands > 0 and MP.GHOST._hand_idx < #MP.GHOST._hands
          and guard < 50 do
        ORIG.advance_hand()
        guard = guard + 1
    end
end

-------------------------------------------------------------------------------
-- resolution (live mode): server rule against the agent's FINAL score
-------------------------------------------------------------------------------

-- -> "won" | "game_over" | "continue", reproducing the original's side-effect
-- contract; never calls win_game() (the exhausted-path caller does that on "won").
function GR.resolve_exhausted(ante, chips)
    local b = GR.buffer[ante]
    local hc, af = to_big(chips), to_big(b.final)
    if hc > af then
        MP.GAME.enemy.lives = MP.GAME.enemy.lives - 1
        GR.emit_result(ante, chips, "agent")
        if MP.GAME.enemy.lives <= 0 then
            MP.GAME.won = true
            GR.match_over = true
            GR.emit("match_end", { winner = "human" })
            return "won"
        end
        MP.GAME.end_pvp = true
        return "continue"
    elseif hc < af then
        if MP.LOBBY.config.gold_on_life_loss then
            MP.GAME.comeback_bonus_given = false
            MP.GAME.comeback_bonus = MP.GAME.comeback_bonus + 1
        end
        MP.GAME.lives = MP.GAME.lives - 1
        MP.UI.ease_lives(-1)
        if MP.LOBBY.config.no_gold_on_round_loss and G.GAME.blind
           and G.GAME.blind.dollars then
            G.GAME.blind.dollars = 0
        end
        GR.emit_result(ante, chips, "human")
        if MP.GAME.lives <= 0 then
            GR.match_over = true
            GR.emit("match_end", { winner = "agent" })
            return "game_over"
        end
        MP.GAME.end_pvp = true
        return "continue"
    else
        -- exact tie: NOBODY loses (server rule; deliberate divergence from the
        -- ghost engine's >=)
        GR.emit_result(ante, chips, nil)
        MP.GAME.end_pvp = true
        return "continue"
    end
end

-------------------------------------------------------------------------------
-- MP.GHOST wraps
-------------------------------------------------------------------------------

MP.GHOST.load = function(replay)
    GR.disarm("abandoned")
    ORIG.load(replay)
    if replay and replay._live then GR.arm(replay) end
end

MP.GHOST.clear = function()
    GR.disarm("abandoned")
    ORIG.clear()
end

MP.GHOST.get_enemy_hands = function(ante)
    if not GR.live then return ORIG.get_enemy_hands(ante) end
    local b = GR.buffer[tonumber(ante) or ante]
    if not b then return {} end
    local out = {}
    for _, h in ipairs(b.hands or {}) do
        out[#out + 1] = { score = numstr(h.score),
                          hands_left = h.hands_left or 0, side = "enemy" }
    end
    if #out == 0 then
        out[1] = { score = numstr(b.final), hands_left = 0, side = "enemy" }
    end
    return out
end

MP.GHOST._start_advance_sequence = function()
    if not GR.live then return ORIG.start_advance_sequence() end
    MP.GHOST._advancing = false      -- our wall-clock reveal replaces the animation loop
end

MP.GHOST.resolve_pvp_mid_hand = function(chips)
    if not GR.live then return ORIG.resolve_pvp_mid_hand(chips) end
    local ante = current_ante()
    GR.ensure_start(ante)
    GR.emit("pvp_hand", { ante = ante, score = numstr(chips),
                          hands_left = G.GAME.current_round.hands_left })
    local b = GR.buffer[ante]
    if not b then return false end
    if GR.revealed_all() and to_big(chips) > to_big(b.final) then
        -- the cut: the agent is exhausted and strictly behind
        MP.GAME.enemy.lives = MP.GAME.enemy.lives - 1
        GR.emit_result(ante, chips, "agent")
        if MP.GAME.enemy.lives <= 0 then
            MP.GAME.won = true
            GR.match_over = true
            GR.emit("match_end", { winner = "human" })
            win_game()               -- the mid-hand path calls win_game() itself
            return true
        end
        MP.GAME.end_pvp = true
        return true
    end
    return false
end

MP.GHOST.resolve_pvp_hands_exhausted = function(chips)
    if not GR.live then return ORIG.resolve_pvp_hands_exhausted(chips) end
    local ante = current_ante()
    GR.ensure_start(ante)
    GR.emit("pvp_hand", { ante = ante, score = numstr(chips), hands_left = 0 })
    if not GR.buffer[ante] then
        -- sidecar data hasn't arrived (only possible in a session's first seconds):
        -- park; Game.update completes the resolution when the buffer fills
        GR.pending_exhausted = { ante = ante, chips = chips }
        return "continue"
    end
    GR.reveal_all()
    return GR.resolve_exhausted(ante, chips)
end

MP.GHOST.resolve_round_fail = function()
    local r = ORIG.resolve_round_fail()
    if GR.live then
        GR.emit("round_fail", { ante = current_ante(), lives = MP.GAME.lives })
        if r == "game_over" then
            GR.match_over = true
            GR.emit("match_end", { winner = "agent" })
        end
    end
    return r
end

-------------------------------------------------------------------------------
-- inbox handling
-------------------------------------------------------------------------------

function GR.handle(ev)
    if ev.e == "agent_nemesis" and ev.ante then
        GR.buffer[tonumber(ev.ante)] = { hands = ev.hands or {}, final = ev.final }
    elseif ev.e == "agent_state" and ev.lives then
        MP.GAME.enemy.lives = ev.lives
    elseif ev.e == "agent_dead" then
        if not GR.match_over then
            GR.match_over = true
            MP.GAME.enemy.lives = 0
            MP.GAME.won = true
            GR.emit("match_end", { winner = "human" })
            win_game()
        end
    elseif ev.e == "hello" then
        if ev.agent_name and MP.GHOST.replay then
            MP.GHOST.replay.nemesis_name = ev.agent_name
        end
    end
end

function GR.poll_inbox()
    if not GR.inbox then return end
    local info = NFS.getInfo(GR.inbox)
    if not info or info.size == GR.inbox_size then return end
    GR.inbox_size = info.size
    local content = NFS.read(GR.inbox)
    if not content then return end
    local n = 0
    for line in content:gmatch("(.-)\n") do        -- complete lines only
        n = n + 1
        if n > GR.inbox_seen then
            GR.inbox_seen = n
            local ok, ev = pcall(json.decode, line)
            if ok and type(ev) == "table" then
                local ok2, err = pcall(GR.handle, ev)
                if not ok2 then
                    sendWarnMessage("GhostRace: handler error: " .. tostring(err),
                                    "GHOSTRACE")
                end
            end
        end
    end
end

-------------------------------------------------------------------------------
-- per-frame driver (the mod's standard Game.update chain hook + wall clock)
-------------------------------------------------------------------------------

function GR.tick_run(now)
    if not (G.GAME and G.GAME.blind and MP.is_pvp_boss()) then return end
    local ante = current_ante()
    if G.STATE == G.STATES.SELECTING_HAND or G.STATE == G.STATES.DRAW_TO_HAND
       or G.STATE == G.STATES.HAND_PLAYED then
        GR.ensure_start(ante)
    end
    -- late init: the blind began before the buffer had this ante
    if #MP.GHOST._hands == 0 and GR.buffer[ante] then
        MP.GHOST.init_playback(ante)
    end
    -- paced reveal: the agent "plays" one hand every REVEAL_S beside the human
    if #MP.GHOST._hands > 0 and MP.GHOST._hand_idx < #MP.GHOST._hands
       and now - GR.last_reveal >= GR.REVEAL_S then
        GR.last_reveal = now
        ORIG.advance_hand()
    end
    -- a parked exhaustion whose data has arrived: finish what the resolver started
    if GR.pending_exhausted and GR.buffer[GR.pending_exhausted.ante] then
        local p = GR.pending_exhausted
        GR.pending_exhausted = nil
        GR.reveal_all()
        local r = GR.resolve_exhausted(p.ante, p.chips)
        if r == "won" then
            win_game()
        elseif r == "game_over" then
            G.STATE = G.STATES.GAME_OVER
            G.STATE_COMPLETE = false
        elseif MP.GAME.end_pvp then          -- replicate the caller's round push
            MP.GAME.end_pvp = false
            G.STATE = G.STATES.NEW_ROUND
            G.STATE_COMPLETE = false
        end
    end
end

local game_update_ref = Game.update
function Game:update(dt)
    game_update_ref(self, dt)
    if not (GR.live and MP.GHOST.is_active()) then return end
    local now = love.timer.getTime()
    if now - GR.last_poll >= GR.POLL_S then
        GR.last_poll = now
        local ok, err = pcall(GR.poll_inbox)
        if not ok then
            sendWarnMessage("GhostRace: poll error: " .. tostring(err), "GHOSTRACE")
        end
        if G.STAGE == G.STAGES.RUN and not GR.match_over then
            local ok2, err2 = pcall(GR.tick_run, now)
            if not ok2 then
                sendWarnMessage("GhostRace: tick error: " .. tostring(err2),
                                "GHOSTRACE")
            end
        end
    end
end

sendDebugMessage("GhostRace loaded (wrapping MP.GHOST for live sessions)", "GHOSTRACE")
