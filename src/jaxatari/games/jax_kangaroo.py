import os
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any, Optional
import jax
import jax.numpy as jnp
import chex
from jax import Array

import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action
from jaxatari.renderers import JAXGameRenderer
import jaxatari.rendering.jax_rendering_utils as jr
from jaxatari.games.kangaroo_levels import (
    LevelConstants,
    Kangaroo_Level_1,
    Kangaroo_Level_2,
    Kangaroo_Level_3,
)

class KangarooConstants(NamedTuple):
    RESET: int = 18
    RENDER_SCALE_FACTOR: int = 4
    SCREEN_WIDTH: int = 160
    SCREEN_HEIGHT: int = 210
    PLAYER_WIDTH: int = 8
    PLAYER_HEIGHT: int = 24
    ENEMY_WIDTH: int = 8
    ENEMY_HEIGHT: int = 24
    FRUIT_WIDTH: int = 8
    FRUIT_HEIGHT: int = 12
    MAX_PLATFORMS: int = 10
    BELL_WIDTH: int = 6
    BELL_HEIGHT: int = 11
    CHILD_WIDTH: int = 8
    CHILD_HEIGHT: int = 15
    MONKEY_WIDTH: int = 6
    MONKEY_HEIGHT: int = 15
    MONKEY_COLOR: Tuple[int, int, int] = (227, 159, 89)
    BACKGROUND_COLOR: Tuple[int, int, int] = (80, 0, 132)
    PLAYER_COLOR: Tuple[int, int, int] = (223, 183, 85)
    ENEMY_COLOR: Tuple[int, int, int] = (227, 151, 89)
    FRUIT_COLOR_STATE_1: Tuple[int, int, int] = (214, 92, 92)
    FRUIT_COLOR_STATE_2: Tuple[int, int, int] = (230, 250, 92)
    FRUIT_COLOR_STATE_3: Tuple[int, int, int] = (255, 92, 250)
    FRUIT_COLOR_STATE_4: Tuple[int, int, int] = (0, 92, 250)
    FRUIT_COLOR: list = [
        (214, 92, 92),
        (230, 250, 92),
        (255, 92, 250),
        (0, 92, 250),
    ]
    COCONUT_COLOR: Tuple[int, int, int] = (162, 98, 33)
    PLATFORM_COLOR: Tuple[int, int, int] = (162, 98, 33)
    LADDER_COLOR: Tuple[int, int, int] = (129, 78, 26)
    BELL_COLOR: Tuple[int, int, int] = (210, 164, 74)
    PLAYER_START_X: int = 23
    PLAYER_START_Y: int = 148
    MOVEMENT_SPEED: int = 1
    LEFT_CLIP: int = 16
    RIGHT_CLIP: int = 144
    COCONUT_WIDTH: int = 2
    COCONUT_HEIGHT: int = 3
    LADDER_HEIGHT: chex.Array = jnp.array(35)
    LADDER_WIDTH: chex.Array = jnp.array(8)
    P_HEIGHT: chex.Array = jnp.array(4)
    LEVEL_1: LevelConstants = Kangaroo_Level_1
    LEVEL_2: LevelConstants = Kangaroo_Level_2
    LEVEL_3: LevelConstants = Kangaroo_Level_3

# -------- Entity Classes --------
class Entity(NamedTuple):
    x: chex.Array
    y: chex.Array
    w: chex.Array
    h: chex.Array


class PlayerState(NamedTuple):
    # Player position
    x: chex.Array
    y: chex.Array
    vel_x: chex.Array
    orientation: chex.Array
    height: chex.Array
    # crouching
    is_crouching: chex.Array
    # jumping
    is_jumping: chex.Array
    jump_base_y: chex.Array
    jump_counter: chex.Array
    jump_orientation: chex.Array
    landing_base_y: chex.Array
    # climbing
    is_climbing: chex.Array
    climb_base_y: chex.Array
    climb_counter: chex.Array
    cooldown_counter: chex.Array
    # other
    is_crashing: chex.Array
    chrash_timer: chex.Array
    punch_left: chex.Array
    punch_right: chex.Array
    last_stood_on_platform_y: chex.Array
    walk_animation: chex.Array
    punch_counter: chex.Array  # New field to track consecutive punches
    needs_release: chex.Array  # New field to track if spacebar needs to be released


class LevelState(NamedTuple):
    """All level related state variables."""

    timer: chex.Array
    platform_positions: chex.Array
    platform_sizes: chex.Array
    ladder_positions: chex.Array
    ladder_sizes: chex.Array
    fruit_positions: chex.Array
    fruit_actives: chex.Array
    fruit_stages: chex.Array
    bell_position: chex.Array
    bell_timer: chex.Array
    child_position: chex.Array
    child_velocity: chex.Array
    child_timer: chex.Array
    falling_coco_position: chex.Array
    falling_coco_dropping: chex.Array
    falling_coco_counter: chex.Array
    falling_coco_skip_update: chex.Array
    step_counter: chex.Array
    monkey_states: chex.Array
    """
    - 0: non-existent
    - 1: moving down
    - 2: moving left
    - 3: throwing
    - 4: moving right
    - 5: moving up
    """
    monkey_positions: chex.Array
    """2D array: [monkey_index, [x, y]]"""
    monkey_throw_timers: chex.Array
    spawn_protection: chex.Array
    coco_positions: chex.Array
    coco_states: chex.Array
    """
    - 0: non existent
    - 1: charging
    - 2: throwing
    """
    spawn_position: chex.Array
    """
    - 0: foot
    - 1: head
    """
    bell_animation: chex.Array


class KangarooState(NamedTuple):
    player: PlayerState
    level: LevelState
    score: chex.Array
    current_level: chex.Array
    level_finished: chex.Array
    levelup_timer: chex.Array
    reset_coords: chex.Array
    levelup: chex.Array
    lives: chex.Array


class KangarooObservation(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    player_o: chex.Array
    platform_positions: chex.Array
    ladder_positions: chex.Array
    fruit_positions: chex.Array
    bell_position: chex.Array
    child_position: chex.Array
    falling_coco_position: chex.Array
    monkey_positions: chex.Array
    coco_positions: chex.Array


class KangarooInfo(NamedTuple):
    score: chex.Array
    level: chex.Array
    all_rewards: chex.Array


class JaxKangaroo(JaxEnvironment[KangarooState, KangarooObservation, KangarooInfo, KangarooConstants]):
    def __init__(self, consts: KangarooConstants = None, frameskip: int = 1, reward_funcs: list[callable]=None):
        super().__init__(consts)
        self.frameskip = frameskip
        self.frame_stack_size = 4
        if reward_funcs is not None:
            reward_funcs = tuple(reward_funcs)
        self.reward_funcs = reward_funcs
        self.action_set = [
            Action.NOOP,
            Action.FIRE,
            Action.UP,
            Action.RIGHT,
            Action.LEFT,
            Action.DOWN,
            Action.UPRIGHT,
            Action.UPLEFT,
            Action.DOWNRIGHT,
            Action.DOWNLEFT,
            Action.UPFIRE,
            Action.RIGHTFIRE,
            Action.LEFTFIRE,
            Action.DOWNFIRE,
            Action.UPRIGHTFIRE,
            Action.UPLEFTFIRE,
            Action.DOWNRIGHTFIRE,
            Action.DOWNLEFTFIRE
        ]
        self.consts = consts or KangarooConstants()
        self.obs_size = 111
        self.renderer = KangarooRenderer(self.consts)

    @partial(jax.jit, static_argnums=(0,))
    def _get_valid_platforms(self, level_constants: LevelConstants) -> chex.Array:
        return level_constants.platform_positions[:, 0] != -1

    @partial(jax.jit, static_argnums=(0, 2), donate_argnums=(1,))
    def _get_platforms_below_player(self, state: KangarooState, y_offset=0) -> chex.Array:
        player_x = state.player.x
        player_y = state.player.y + y_offset
        player_bottom_y = player_y + state.player.height

        level_constants: LevelConstants = self._get_level_constants(state.current_level)

        platform_positions = level_constants.platform_positions
        platform_sizes = level_constants.platform_sizes

        platform_x = platform_positions[:, 0]
        platform_y = platform_positions[:, 1]
        platform_width = platform_sizes[:, 0]

        player_is_within_platform_x = jnp.logical_and(
            (player_x + self.consts.PLAYER_WIDTH) >= platform_x,
            player_x <= (platform_x + platform_width),
        )

        platform_is_below_player = player_bottom_y <= platform_y

        diff_to_platforms = jnp.where(
            platform_is_below_player, platform_y - player_bottom_y, 1000
        )

        valid_platforms = self._get_valid_platforms(level_constants)

        candidate_platforms = (
            player_is_within_platform_x & platform_is_below_player & valid_platforms
        )

        masked_diffs = jnp.where(candidate_platforms, diff_to_platforms, 1000)

        closest_platform_idx = jnp.argmin(masked_diffs)
        min_diff = masked_diffs[closest_platform_idx]

        result = jnp.zeros_like(platform_x, dtype=bool)
        return result.at[closest_platform_idx].set(min_diff < 1000)


    @partial(jax.jit, static_argnums=(0,))
    def _entities_collide_with_threshold(
        self,
        e1_x: chex.Array,
        e1_y: chex.Array,
        e1_w: chex.Array,
        e1_h: chex.Array,
        e2_x: chex.Array,
        e2_y: chex.Array,
        e2_w: chex.Array,
        e2_h: chex.Array,
        threshold: chex.Array,
    ) -> chex.Array:
        """Returns True if rectangles overlap by at least threshold fraction. This only Checks for overlap in the x dimension.
        e1_x, e1_y, e1_w, e1_h: Entity 1 position and size
        e2_x, e2_y, e2_w, e2_h: Entity 2 position and size
        threshold: Minimum fraction of overlap required (0-1)

        Returns:
            bool: True if entities overlap by at least threshold fraction, False otherwise
        """
        overlap_start_x = jnp.maximum(e1_x, e2_x)
        overlap_end_x = jnp.minimum(e1_x + e1_w, e2_x + e2_w)
        overlap_start_y = jnp.maximum(e1_y, e2_y)
        overlap_end_y = jnp.minimum(e1_y + e1_h, e2_y + e2_h)

        # Calculate dimensions of overlap region
        overlap_width = overlap_end_x - overlap_start_x
        overlap_height = overlap_end_y - overlap_start_y

        smallest_entity_width = jnp.minimum(e1_w, e2_w)

        # Calculate minimum required overlap area based on threshold
        min_required_overlap = smallest_entity_width * threshold

        meets_threshold = overlap_width >= min_required_overlap

        return jnp.where((overlap_width < 0) | (overlap_height < 0), False, meets_threshold)

    @partial(jax.jit, static_argnums=(0,))
    def _entities_collide(self, e1_x: chex.Array, e1_y: chex.Array, e1_w: chex.Array, e1_h: chex.Array, e2_x: chex.Array, e2_y: chex.Array, e2_w: chex.Array, e2_h: chex.Array) -> chex.Array:
        return self._entities_collide_with_threshold(
            e1_x, e1_y, e1_w, e1_h, e2_x, e2_y, e2_w, e2_h, 0
        )

    @partial(jax.jit, static_argnums=(0, 2, 3), donate_argnums=(1,))
    def _player_is_above_ladder(self, state: KangarooState, threshold: float = 0.3, virtual_hitbox_height: float = 12.0) -> chex.Array:
        level_constants: LevelConstants = self._get_level_constants(state.current_level)

        ladder_x = level_constants.ladder_positions[:, 0]
        ladder_y = level_constants.ladder_positions[:, 1]
        ladder_w = level_constants.ladder_sizes[:, 0]
        ladder_h = level_constants.ladder_sizes[:, 1]

        return jax.vmap(
            self._entities_collide_with_threshold,
            in_axes=(None, None, None, None, 0, 0, 0, 0, None),
        )(
            state.player.x,
            state.player.y + state.player.height,
            self.consts.PLAYER_WIDTH,
            virtual_hitbox_height,
            ladder_x,
            ladder_y,
            ladder_w,
            ladder_h,
            threshold,
        )

    @partial(jax.jit, static_argnums=(0, 2), donate_argnums=(1,))
    def _check_ladder_collisions(self, state: KangarooState, threshold: float = 0.3) -> chex.Array:
        level_constants: LevelConstants = self._get_level_constants(state.current_level)

        ladder_x = level_constants.ladder_positions[:, 0]
        ladder_y = level_constants.ladder_positions[:, 1]
        ladder_w = level_constants.ladder_sizes[:, 0]
        ladder_h = level_constants.ladder_sizes[:, 1]

        return jax.vmap(
            self._entities_collide_with_threshold,
            in_axes=(None, None, None, None, 0, 0, 0, 0, None),
        )(
            state.player.x,
            state.player.y + 16,
            self.consts.PLAYER_WIDTH,
            state.player.height - 16,
            ladder_x,
            ladder_y,
            ladder_w,
            ladder_h,
            threshold,
        )

    @partial(jax.jit, static_argnums=(0, 4), donate_argnums=(1,))
    def _player_is_on_ladder(self, state: KangarooState, ladder_pos: chex.Array, ladder_size: chex.Array, threshold: float = 0.3) -> chex.Array:
        return self._entities_collide_with_threshold(
            state.player.x,
            state.player.y,
            self.consts.PLAYER_WIDTH,
            state.player.height,
            ladder_pos[0],
            ladder_pos[1],
            ladder_size[0],
            ladder_size[1],
            threshold,
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _player_jump_controller(self, state: KangarooState, jump_pressed: chex.Array, ladder_intersect: chex.Array):
        player_y = state.player.y
        jump_counter = state.player.jump_counter
        is_jumping = state.player.is_jumping

        cooldown_condition = state.player.cooldown_counter > 0
        jump_start = (
            jump_pressed
            & ~is_jumping
            & ~ladder_intersect
            & ~cooldown_condition
            & ((player_y + self.consts.PLAYER_HEIGHT) > 28)
        )

        jump_counter = jnp.where(jump_start, 0, jump_counter)
        jump_orientation = jnp.where(
            jump_start, state.player.orientation, state.player.jump_orientation
        )
        jump_base_y = jnp.where(jump_start, player_y, state.player.jump_base_y)
        new_landing_base_y = jump_base_y

        platform_y_below_player = self._get_y_of_platform_below_player(state)

        # find a new potential landing_base if player is above a higher platform
        new_landing_base_y = jnp.where(
            is_jumping
            & ((platform_y_below_player - self.consts.PLAYER_HEIGHT) == (jump_base_y - 8))
            & ~jump_start,
            platform_y_below_player - self.consts.PLAYER_HEIGHT,
            new_landing_base_y,
        )

        # --- Allow jumping down: if the player reaches a platform located exactly
        # --- 8 pixels below the jump base, treat it as a valid landing base too.
        new_landing_base_y = jnp.where(
            is_jumping
            & ((platform_y_below_player - self.consts.PLAYER_HEIGHT) == (jump_base_y + 8))
            & ~jump_start,
            platform_y_below_player - self.consts.PLAYER_HEIGHT,
            new_landing_base_y,
        )

        is_jumping = is_jumping | jump_start

        jump_counter = jnp.where(is_jumping, jump_counter + 1, jump_counter)

        # Calculate vertical offset based on jump phase
        def offset_for(count):
            conditions = [
                (count > 0) & (count <= 8),
                (count > 8) & (count < 16),
                (count >= 16) & (count <= 24),
                (count > 24) & (count <= 32),
                (count > 32) & (count < 40),
            ]
            values = [
                -1,
                -8,
                -8,
                -16,
                -8,
            ]
            return jnp.select(conditions, values, default=0)

        # check if player is on a new platform and cancel jump if so
        jump_cancel_up = (
            is_jumping
            & (player_y >= new_landing_base_y)
            & (new_landing_base_y < jump_base_y)
            & (jump_counter > 32)
        )

        jump_cancel_down = (
            is_jumping
            & ((player_y + 1) == jump_base_y)  # +1 because for some reason the player is 1 pixel below the value I would expect
            & (new_landing_base_y == (jump_base_y + 8))
            & (jump_counter >= 40)
        )

        jump_cancel = jump_cancel_up | jump_cancel_down

        jump_counter = jnp.where(jump_cancel, 40, jump_counter)
        jump_base_y = jnp.where(jump_cancel, new_landing_base_y, jump_base_y)
        new_y = jnp.where(jump_cancel, new_landing_base_y, player_y)
        new_cooldown_counter = jnp.where(jump_cancel, 8, state.player.cooldown_counter)

        total_offset = offset_for(jump_counter)
        new_y = jnp.where(is_jumping & ~jump_cancel, jump_base_y + total_offset, new_y)

        jump_complete = jump_counter >= 41
        is_jumping = jnp.where(jump_complete, False, is_jumping)
        jump_counter = jnp.where(jump_complete, 0, jump_counter)

        return_value = (
            new_y,
            jump_counter,
            is_jumping,
            jump_base_y,
            new_landing_base_y,
            jump_orientation,
            new_cooldown_counter,
        )

        return jax.lax.cond(
            state.levelup_timer == 0,
            lambda: return_value,
            lambda: (
                state.player.y,
                state.player.jump_counter,
                state.player.is_jumping,
                state.player.jump_base_y,
                state.player.landing_base_y,
                state.player.jump_orientation,
                state.player.cooldown_counter,
            ),
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _player_climb_controller(self, state: KangarooState, y: chex.Array, press_up: chex.Array, press_down: chex.Array, ladder_intersect: chex.Array) -> tuple[Array, Array, Array, Array, Array]:
        ladder_intersect_below = jnp.any(self._player_is_above_ladder(state))

        new_y = y
        is_climbing = state.player.is_climbing
        is_climbing = jnp.where(state.player.is_jumping, False, is_climbing)

        climb_counter = state.player.climb_counter

        cooldown_over = state.player.cooldown_counter <= 0

        climb_start = (
            press_up
            & ~is_climbing
            & ladder_intersect
            & ~state.player.is_jumping
            & cooldown_over
        )
        climb_start_downward = (
            press_down
            & ~is_climbing
            & ladder_intersect_below
            & ~state.player.is_jumping
            & cooldown_over
        )

        is_climbing = is_climbing | climb_start | climb_start_downward

        climb_counter = jnp.where(climb_start | climb_start_downward, 0, climb_counter)

        climb_base_y = jnp.where(climb_start, new_y, state.player.climb_base_y)

        climb_base_y = jnp.where(
            climb_start_downward,
            self._get_y_of_platform_below_player(state, 1) - self.consts.PLAYER_HEIGHT,
            climb_base_y,
        )

        new_y = jnp.where(climb_start, new_y - 8, new_y)
        new_y = jnp.where(climb_start_downward, new_y + 8, new_y)

        climb_counter = jnp.where(is_climbing, climb_counter + 1, climb_counter)

        climb_up = jnp.logical_and(press_up, is_climbing)
        climb_down = jnp.logical_and(press_down, is_climbing)

        new_y = jnp.where(
            jnp.logical_and(climb_up, jnp.equal(climb_counter, 19)), new_y - 8, new_y
        )
        new_y = jnp.where(
            jnp.logical_and(climb_down, jnp.equal(climb_counter, 19)), new_y + 8, new_y
        )

        set_new_climb_base = (
            climb_up
            & ((self._get_y_of_platform_below_player(state) - state.player.height) >= new_y)
            & ladder_intersect
        )
        climb_base_y = jnp.where(
            set_new_climb_base,
            self._get_y_of_platform_below_player(state) - self.consts.PLAYER_HEIGHT,
            climb_base_y,
        )
        climb_stop = is_climbing & (new_y >= climb_base_y) & ~climb_start_downward

        is_climbing = jnp.where(climb_stop, False, is_climbing)

        is_climbing = jnp.where(ladder_intersect | climb_start_downward, is_climbing, False)

        climb_counter = jnp.where(climb_counter >= 19, 0, climb_counter)
        cooldown_counter = jnp.where(
            climb_stop | set_new_climb_base,
            15,
            jnp.where(
                state.player.cooldown_counter > 0, state.player.cooldown_counter - 1, 0
            ),
        )

        return_value = (new_y, is_climbing, climb_base_y, climb_counter, cooldown_counter)

        return jax.lax.cond(
            state.levelup_timer == 0,
            lambda: return_value,
            lambda: (
                state.player.y,
                state.player.is_climbing,
                state.player.climb_base_y,
                state.player.climb_counter,
                state.player.cooldown_counter,
            ),
        )

    @partial(jax.jit, static_argnums=(0,))
    def _player_height_controller(self, is_jumping: chex.Array, jump_counter: chex.Array, is_crouching: chex.Array) -> chex.Array:
        def jump_based_height(count):
            conditions = [
                (count < 8),
                (count < 16),
                (count < 24),
                (count < 40),
            ]
            values = [
                23,
                24,
                15,
                23,
            ]
            return jnp.select(conditions, values, default=24)

        candidate_height = jump_based_height(jump_counter)
        height_if_jumping = jnp.where(is_jumping, candidate_height, 24)

        is_crouching = jnp.logical_and(is_crouching, jnp.logical_not(is_jumping))

        new_height = jnp.where(is_crouching, 16, height_if_jumping)
        return new_height

    @partial(jax.jit, static_argnums=(0, 2), donate_argnums=(1,))
    def _get_y_of_platform_below_player(self, state: KangarooState, y_offset=0) -> chex.Array:
        level_constants: LevelConstants = self._get_level_constants(state.current_level)

        platform_bands: jax.Array = self._get_platforms_below_player(state, y_offset)
        platform_ys = level_constants.platform_positions[:, 1]

        has_platform_below = jnp.any(platform_bands)

        platform_y = jnp.sum(platform_bands * platform_ys)

        return jnp.where(has_platform_below, platform_y, jnp.array(1000))

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _fruits_step(self, state: KangarooState) -> Tuple[chex.Array, chex.Array]:
        fruit_x = state.level.fruit_positions[:, 0]
        fruit_y = state.level.fruit_positions[:, 1]

        def check_fruit(p_x, p_y, p_w, p_h, f_x, f_y, f_w, f_h, stage, active):
            fruit_collision = self._entities_collide(p_x, p_y, p_w, p_h, f_x, f_y, f_w, f_h)
            collision_condition = jnp.logical_and(fruit_collision, active)
            return jnp.where(collision_condition, 100 * (2**stage), 0), jnp.where(
                collision_condition, False, active
            )

        (score_additions, new_activations) = jax.vmap(
            check_fruit, in_axes=(None, None, None, None, 0, 0, None, None, 0, 0)
        )(
            state.player.x,
            state.player.y,
            self.consts.PLAYER_WIDTH,
            state.player.height,
            fruit_x,
            fruit_y,
            self.consts.FRUIT_WIDTH,
            self.consts.FRUIT_HEIGHT,
            state.level.fruit_stages,
            state.level.fruit_actives,
        )
        new_score = jnp.sum(score_additions)

        bell_collision = self._entities_collide(
            state.player.x,
            state.player.y,
            self.consts.PLAYER_WIDTH,
            state.player.height,
            state.level.bell_position[0],
            state.level.bell_position[1],
            self.consts.BELL_WIDTH,
            self.consts.BELL_HEIGHT,
        )
        bell_active = ~jnp.any(state.level.fruit_stages == 3)

        RESPAWN_AFTER_TICKS = 40

        counter = state.level.bell_timer
        counter_start = bell_collision & (counter == 0) & bell_active
        counter = jnp.where(counter_start, 1, counter)
        counter = jnp.where(counter > 0, counter + 1, counter)
        counter = jnp.where(counter == RESPAWN_AFTER_TICKS + 1, 0, counter)
        respawn_timer_done = counter == RESPAWN_AFTER_TICKS

        def get_new_stages(respawn_timer_done, active, stage):
            return jnp.where(
                respawn_timer_done & (~active),
                jnp.clip(stage + 1, 0, 3),
                stage,
            )

        new_stages = jax.vmap(get_new_stages, in_axes=(None, 0, 0))(
            respawn_timer_done, state.level.fruit_actives, state.level.fruit_stages
        )

        activations = jax.lax.cond(
            respawn_timer_done,
            lambda: jnp.less_equal(new_stages, jnp.array([3, 3, 3])),
            lambda: new_activations,
        )

        return new_score, activations, new_stages, counter

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _child_step(self, state: KangarooState) -> Tuple[chex.Array]:
        RESET_TIMER_AFTER = 50

        counter = state.level.child_timer
        counter = counter + 1
        counter = jnp.where(counter > RESET_TIMER_AFTER, 0, counter)
        reset = counter == RESET_TIMER_AFTER

        child_velocity = state.level.child_velocity
        new_child_velocity = jnp.where(reset, child_velocity * -1, child_velocity)

        new_child_x = jnp.where(
            state.levelup_timer == 0,
            jnp.where(
                (counter % 5) == 0,
                state.level.child_position[0] + new_child_velocity,
                state.level.child_position[0],
            ),
            state.level.child_position[0],
        )
        new_child_y = state.level.child_position[1]
        new_child_timer = counter

        return new_child_timer, new_child_x, new_child_y, new_child_velocity

    def _pad_array(self, arr: jax.Array, target_size: int):
        current_size = arr.shape[0]

        return jnp.pad(
            arr,
            ((0, target_size - current_size), (0, 0)),
            mode="constant",
            constant_values=-1,
        )

    def _pad_to_size(self, level_constants: LevelConstants, max_platforms: int):
        return LevelConstants(
            ladder_positions=self._pad_array(level_constants.ladder_positions, max_platforms),
            ladder_sizes=self._pad_array(level_constants.ladder_sizes, max_platforms),
            platform_positions=self._pad_array(level_constants.platform_positions, max_platforms),
            platform_sizes=self._pad_array(level_constants.platform_sizes, max_platforms),
            fruit_positions=level_constants.fruit_positions,
            bell_position=level_constants.bell_position,
            child_position=level_constants.child_position,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_level_constants(self, current_level: int) -> LevelConstants:
        max_platforms = 20

        level1_padded = self._pad_to_size(self.consts.LEVEL_1, max_platforms)
        level2_padded = self._pad_to_size(self.consts.LEVEL_2, max_platforms)
        level3_padded = self._pad_to_size(self.consts.LEVEL_3, max_platforms)

        return jax.lax.cond(
            current_level == 1,
            lambda _: level1_padded,
            lambda _: jax.lax.cond(
                current_level == 2,
                lambda _: level2_padded,
                lambda _: level3_padded,
                operand=None,
            ),
            operand=None,
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _player_step(self, state: KangarooState, action: chex.Array):
        level_constants = self._get_level_constants(state.current_level)
        x, y = state.player.x, state.player.y
        old_height = state.player.height
        old_orientation = state.player.orientation

        # Get inputs
        press_right = jnp.any(
            jnp.array(
                [
                    action == Action.RIGHT,
                    action == Action.UPRIGHT,
                    action == Action.DOWNRIGHT,
                ]
            )
        )

        press_left = jnp.any(
            jnp.array(
                [action == Action.LEFT, action == Action.UPLEFT, action == Action.DOWNLEFT]
            )
        )

        press_up = jnp.any(
            jnp.array(
                [action == Action.UP, action == Action.UPRIGHT, action == Action.UPLEFT]
            )
        )

        # Store original fire press state before any modifications
        original_press_fire = jnp.any(
            jnp.array(
                [
                    action == Action.FIRE,
                    action == Action.DOWNFIRE,
                    action == Action.UPLEFTFIRE,
                    action == Action.UPRIGHTFIRE,
                    action == Action.DOWNLEFTFIRE,
                    action == Action.DOWNRIGHTFIRE,
                ]
            )
        )

        press_down = jnp.any(
            jnp.array(
                [
                    action == Action.DOWN,
                    action == Action.DOWNLEFT,
                    action == Action.DOWNRIGHT,
                ]
            )
        )

        press_down = jnp.where(state.player.is_jumping, False, press_down)
        original_press_fire = jnp.where(state.player.is_jumping, False, original_press_fire)
        original_press_fire = jnp.where(
            state.player.is_climbing, False, original_press_fire
        )

        press_up = jnp.where(press_down, False, press_up)

        press_right = jnp.where(state.player.is_climbing, False, press_right)
        press_left = jnp.where(state.player.is_climbing, False, press_left)

        is_looking_left = state.player.orientation == -1
        is_looking_right = state.player.orientation == 1

        # Update punch counter
        new_punch_counter = jnp.where(
            original_press_fire, state.player.punch_counter + 1, state.player.punch_counter
        )

        # Reset counter when fire is released
        new_punch_counter = jnp.where(
            ~original_press_fire & (state.player.punch_counter > 0), 0, new_punch_counter
        )

        # Set needs_release flag when counter reaches 28 and keep it true until spacebar is released
        new_needs_release = jnp.where(
            new_punch_counter >= 28,
            True,  # Need to release spacebar
            jnp.where(
                ~original_press_fire,  # If spacebar is released
                False,  # Reset the flag
                state.player.needs_release,  # Otherwise keep current state
            ),
        )

        # Only allow punching if either:
        # 1. Counter is below 28, or
        # 2. Spacebar has been released after hitting 28
        can_punch = jnp.logical_or(new_punch_counter < 28, ~new_needs_release)

        press_fire = jnp.where(can_punch, original_press_fire, False)

        is_punching_left = (
            jnp.logical_and(press_fire, is_looking_left) & ~state.player.is_crashing
        )
        is_punching_right = (
            jnp.logical_and(press_fire, is_looking_right) & ~state.player.is_crashing
        )

        ladder_intersect_thresh = jnp.any(self._check_ladder_collisions(state))
        ladder_intersect_no_thresh = jnp.any(self._check_ladder_collisions(state, 0))

        ladder_intersect = jnp.where(
            state.player.is_climbing, ladder_intersect_no_thresh, ladder_intersect_thresh
        )

        (
            new_y,
            new_jump_counter,
            new_is_jumping,
            new_jump_base_y,
            new_landing_base_y,
            new_jump_orientation,
            new_cooldown_counter,
        ) = self._player_jump_controller(state, press_up, ladder_intersect)

        (
            new_y,
            new_is_climbing,
            new_climb_base_y,
            new_climb_counter,
            new_cooldown_counter,
        ) = self._player_climb_controller(state, new_y, press_up, press_down, ladder_intersect)

        new_is_crouching = press_down & ~new_is_climbing & ~new_is_jumping

        candidate_vel_x = jnp.where(
            new_is_crouching,
            0,
            jnp.where(
                press_left, -self.consts.MOVEMENT_SPEED, jnp.where(press_right, self.consts.MOVEMENT_SPEED, 0)
            ),
        )

        standing_still = jnp.equal(candidate_vel_x, 0)
        new_orientation = jnp.sign(candidate_vel_x)
        new_orientation = jnp.where(standing_still, old_orientation, new_orientation)

        stop_in_air = jnp.logical_and(
            new_is_jumping, state.player.jump_orientation != new_orientation
        )

        vel_x = jnp.where(stop_in_air, 0, candidate_vel_x)

        new_player_height = self._player_height_controller(
            is_jumping=new_is_jumping,
            jump_counter=new_jump_counter,
            is_crouching=new_is_crouching,
        )
        new_player_height = jnp.where(
            (state.levelup_timer > 0) | state.player.is_crashing,
            self.consts.PLAYER_HEIGHT,
            new_player_height,
        )
        dy = old_height - new_player_height
        new_y = new_y + dy

        # x-axis movement
        x = jnp.where(
            state.level.step_counter % 3 != 0,
            x,
            jnp.where(
                state.player.is_crashing | state.levelup_timer != 0,
                x,
                jnp.clip(x + vel_x, self.consts.LEFT_CLIP, self.consts.RIGHT_CLIP - self.consts.PLAYER_WIDTH),
            ),
        )

        platform_bools: jax.Array = self._get_platforms_below_player(state)
        platform_ys: jax.Array = level_constants.platform_positions[:, 1]

        valid_platforms = self._get_valid_platforms(level_constants)

        valid_and_affecting = jnp.logical_and(platform_bools, valid_platforms)

        climbing_transition = ~state.player.is_climbing & new_is_climbing & press_down

        # For each platform, calculate what y would be if player is positioned on it
        platform_y_values = jnp.where(
            climbing_transition | state.player.is_jumping, new_y, jnp.clip(new_y, 0, platform_ys - new_player_height)
        )

        masked_platform_y_values = jnp.where(valid_and_affecting, platform_y_values, new_y)

        platform_dependent_y = jnp.min(
            jnp.where(valid_and_affecting, masked_platform_y_values, self.consts.SCREEN_HEIGHT)
        )

        y = jnp.where(
            state.player.is_crashing,
            jnp.where((y + new_player_height) > self.consts.SCREEN_HEIGHT, y, y + 2),
            platform_dependent_y,
        )

        final_platform_y = 28
        player_on_last_platform = (new_y + new_player_height) == final_platform_y
        level_finished = (
            player_on_last_platform & ~state.level_finished & (state.levelup_timer == 0)
        )

        y = jnp.where(state.levelup_timer == 0, y, state.player.y)

        x = jnp.where(state.reset_coords, self.consts.PLAYER_START_X, x)
        y = jnp.where(state.reset_coords, self.consts.PLAYER_START_Y, y)

        return (
            x,
            y,
            vel_x,
            new_is_crouching,
            new_is_jumping,
            new_is_climbing,
            new_jump_counter,
            new_orientation,
            new_jump_base_y,
            new_landing_base_y,
            new_player_height,
            new_jump_orientation,
            new_climb_base_y,
            new_climb_counter,
            is_punching_left,
            is_punching_right,
            new_cooldown_counter,
            level_finished,
            new_punch_counter,
            new_needs_release,
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _timer_controller(self, state: KangarooState):
        return jnp.where(
            state.level.step_counter == 255, state.level.timer - 100, state.level.timer
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _next_level(self, state: KangarooState):
        RESET_AFTER_TICKS = 256

        counter = state.levelup_timer
        counter_start = state.level_finished & (counter == 0)
        counter = jnp.where((counter > 0) | counter_start, counter + 1, counter)
        reset_timer_done = counter == RESET_AFTER_TICKS
        counter = jnp.where(counter > RESET_AFTER_TICKS, 0, counter)

        reset_coords = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))
        levelup = jnp.where(reset_timer_done, jnp.array(True), jnp.array(False))

        current_level = jnp.where(levelup, state.current_level + 1, state.current_level)

        return current_level, counter, reset_coords, levelup

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _lives_controller(self, state: KangarooState):
        is_time_over = state.level.timer <= 0

        new_last_stood_on_platform_y = jnp.where(
            self._get_y_of_platform_below_player(state) == (state.player.y + state.player.height),
            self._get_y_of_platform_below_player(state),
            state.player.last_stood_on_platform_y,
        )

        # platform_drop_check()

        y_of_platform_below_player = self._get_y_of_platform_below_player(state)
        player_is_falling = (
            (state.player.y + state.player.height) == state.player.last_stood_on_platform_y
        ) & (y_of_platform_below_player > state.player.last_stood_on_platform_y) & (~state.player.is_jumping)

        # monkey touch check
        def check_monkey_collision(p_x, p_y, p_w, p_h, m_x, m_y, m_w, m_h, m_state):
            # Add a small delay before re-enabling collision detection
            # Only check collision if monkey state is not 0 and not in the process of being punched
            return jnp.logical_and(
                self._entities_collide(p_x, p_y, p_w, p_h, m_x, m_y, m_w, m_h),
                jnp.logical_and(
                    m_state != 0,
                    jnp.logical_not(
                        jnp.logical_and(
                            m_state == 0,
                            jnp.logical_and(
                                m_x == 152, m_y == 5  # If monkey is at spawn position
                            ),
                        )
                    ),
                ),
            )

        monkey_collision = jax.vmap(
            check_monkey_collision,
            in_axes=(None, None, None, None, 0, 0, None, None, 0),
        )(
            state.player.x,
            state.player.y,
            self.consts.PLAYER_WIDTH,
            state.player.height,
            state.level.monkey_positions[:, 0],
            state.level.monkey_positions[:, 1],
            self.consts.MONKEY_WIDTH,
            self.consts.MONKEY_HEIGHT - 1,
            state.level.monkey_states,
        )

        player_collided_with_monkey = jnp.any(monkey_collision)

        def check_collision(p_x, p_y, p_w, p_h, m_x, m_y, m_w, m_h, m_state):
            return jnp.logical_and(
                self._entities_collide(p_x, p_y, p_w, p_h, m_x, m_y, m_w, m_h - 1), m_state != 0
            )

        collision = jax.vmap(
            check_collision,
            in_axes=(None, None, None, None, 0, 0, None, None, 0),
        )(
            state.player.x,
            state.player.y,
            self.consts.PLAYER_WIDTH,
            state.player.height,
            state.level.coco_positions[:, 0],
            state.level.coco_positions[:, 1],
            self.consts.COCONUT_WIDTH,
            self.consts.COCONUT_HEIGHT,
            state.level.coco_states,
        )

        player_collided_with_horizontal_coco = jnp.any(collision)

        crashed_falling_coco = self._entities_collide_with_threshold(
            state.player.x,
            state.player.y,
            self.consts.PLAYER_WIDTH,
            state.player.height - 8,
            state.level.falling_coco_position[0],
            state.level.falling_coco_position[1],
            self.consts.COCONUT_WIDTH,
            self.consts.COCONUT_HEIGHT,
            0.1,
        )

        remove_live = (
            is_time_over
            | player_is_falling
            | crashed_falling_coco
            | player_collided_with_monkey
            | player_collided_with_horizontal_coco
        ) & ~state.player.is_crashing
        new_is_crashing = jnp.where(remove_live, True, state.player.is_crashing)

        start_timer = (
            state.player.is_crashing
            & (state.player.chrash_timer == 0)
            & ((state.player.y + state.player.height) > self.consts.SCREEN_HEIGHT)
        )

        RESPAWN_AFTER_TICKS = 40

        counter = state.player.chrash_timer
        counter_start = start_timer
        counter = jnp.where(counter_start, 1, counter)
        counter = jnp.where(counter > 0, counter + 1, counter)
        counter = jnp.where(counter == RESPAWN_AFTER_TICKS + 1, 0, counter)
        crash_timer_done = counter == RESPAWN_AFTER_TICKS

        new_is_crashing = jnp.where(crash_timer_done, False, new_is_crashing)

        return (
            jnp.where(remove_live, state.lives - 1, state.lives),
            new_is_crashing,
            counter,
            crash_timer_done,
            new_last_stood_on_platform_y,
        )


    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _falling_coconut_controller(self, state: KangarooState, punching: chex.Array):
        falling_coco_exists = (state.level.falling_coco_position[0] != 13) | (
            state.level.falling_coco_position[1] != -1
        )

        spawn_new_coco = ~falling_coco_exists & (state.level.step_counter == 255)

        update_positions = ~state.level.falling_coco_skip_update & (
            ((state.level.step_counter % 8) == 0) | spawn_new_coco
        )

        # coco go down or up before dropping
        coco_down = (state.level.step_counter % 32) < 16

        # detect if coco is above player and switch from x-following state to dropping state
        coco_first_time_above_player = (
            ~state.level.falling_coco_dropping
            & falling_coco_exists
            & (
                ((state.level.falling_coco_position[0] + self.consts.COCONUT_WIDTH) > state.player.x)
                & (state.level.falling_coco_position[0] < (state.player.x + self.consts.PLAYER_WIDTH))
            )
            & update_positions
        )
        update_positions = jnp.where(coco_first_time_above_player, True, update_positions)

        new_falling_coco_dropping = jnp.where(
            coco_first_time_above_player, True, state.level.falling_coco_dropping
        )

        new_falling_coco_skip_update = coco_first_time_above_player

        new_falling_coco_skip_update = jnp.where(
            state.level.falling_coco_skip_update
            & (((state.level.step_counter % 8) == 0) | spawn_new_coco),
            False,
            new_falling_coco_skip_update | state.level.falling_coco_skip_update,
        )

        new_falling_coco_counter = jnp.where(
            update_positions,
            jnp.where(
                spawn_new_coco,
                0,
                jnp.where(
                    state.level.falling_coco_dropping
                    & update_positions,  # coco is dropping
                    state.level.falling_coco_counter + 1,
                    jnp.where(
                        update_positions & coco_down,  # coco is going down
                        state.level.falling_coco_counter + 1,
                        state.level.falling_coco_counter - 1,
                    ),
                ),
            ),
            state.level.falling_coco_counter,
        )

        # detect if player is punching the coco
        fist_w = 3
        fist_h = 4
        fist_x = jnp.where(
            state.player.orientation > 0,
            state.player.x + self.consts.PLAYER_WIDTH,
            state.player.x - fist_w,
        )
        fist_y = state.player.y + 8

        coco_punching = (
            self._entities_collide_with_threshold(
                fist_x,
                fist_y,
                fist_w,
                fist_h,
                state.level.falling_coco_position[0],
                state.level.falling_coco_position[1],
                self.consts.COCONUT_WIDTH,
                self.consts.COCONUT_HEIGHT,
                0.01,
            )
            & punching
        )

        score_addition = jnp.where(coco_punching, 200, 0)

        reset_coco = (
            (new_falling_coco_counter > 20) | state.player.is_crashing | coco_punching
        )
        new_falling_coco_counter = jnp.where(reset_coco, 0, new_falling_coco_counter)

        new_falling_coco_dropping = jnp.where(reset_coco, False, new_falling_coco_dropping)

        new_falling_coco_position_x = jnp.where(
            update_positions
            & ~state.level.falling_coco_dropping
            & (falling_coco_exists | spawn_new_coco),
            state.level.falling_coco_position[0] + 2,
            state.level.falling_coco_position[0],
        )

        new_falling_coco_position_y = jnp.where(
            update_positions & (falling_coco_exists | spawn_new_coco),
            8 * new_falling_coco_counter + 9,
            state.level.falling_coco_position[1],
        )

        new_falling_coco_position = jnp.where(
            reset_coco,
            jnp.array([13, -1]),
            jnp.array([new_falling_coco_position_x, new_falling_coco_position_y]),
        )

        return (
            new_falling_coco_position,
            new_falling_coco_dropping,
            new_falling_coco_counter,
            new_falling_coco_skip_update,
            score_addition,
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _monkey_controller(self, state: KangarooState, punching: chex.Array):
        current_monkeys_existing = jnp.sum(state.level.monkey_states != 0)

        spawn_new_monkey = (
            ~state.level.spawn_protection
            & (current_monkeys_existing < 4)
            & (state.level.step_counter == 16)
        )

        monkey_states_is_zero = state.level.monkey_states == 0
        first_non_existing_monkey_index = jnp.argmin(~monkey_states_is_zero)
        first_non_existing_monkey_index = jnp.where(
            jnp.any(monkey_states_is_zero), first_non_existing_monkey_index, jnp.array(-1)
        )

        new_monkey_states = state.level.monkey_states
        new_monkey_states = jax.lax.cond(
            spawn_new_monkey,
            lambda: new_monkey_states.at[first_non_existing_monkey_index].set(1),
            lambda: new_monkey_states,
        )

        monkey_lower_y = state.level.monkey_positions[:, 1] + self.consts.MONKEY_HEIGHT
        monkey_on_p1 = monkey_lower_y == 172
        monkey_on_p2 = monkey_lower_y == 124
        monkey_on_p3 = monkey_lower_y == 76

        platform_y_under_player = self._get_y_of_platform_below_player(state)

        transition_1_to_2 = (
            (
                monkey_on_p1
                & (platform_y_under_player <= 172)
                & (platform_y_under_player > 124)
            )
            | (
                monkey_on_p2
                & (platform_y_under_player <= 124)
                & (platform_y_under_player > 76)
            )
            | (
                monkey_on_p3
                & (platform_y_under_player <= 76)
                & (platform_y_under_player > 28)
            )
        )

        new_monkey_states = jnp.where(
            (new_monkey_states == 1) & transition_1_to_2, 2, new_monkey_states
        )

        in_state_1 = new_monkey_states == 1
        should_transition = (state.level.monkey_positions[:, 1] + self.consts.MONKEY_HEIGHT) >= 172
        new_monkey_states = jnp.where(
            in_state_1 & should_transition,
            5,
            new_monkey_states,
        )

        in_state_2 = new_monkey_states == 2
        monkey_x_positions = state.level.monkey_positions[:, 0]
        min_x_reached = monkey_x_positions <= 107
        new_monkey_states = jnp.where(
            in_state_2 & min_x_reached,
            3,
            new_monkey_states,
        )

        in_state_3 = new_monkey_states == 3
        timer_is_zero = state.level.monkey_throw_timers == 0
        should_transition = in_state_3 & timer_is_zero & (state.level.monkey_states == 3)
        new_monkey_states = jnp.where(should_transition, 4, new_monkey_states)

        in_state_4 = new_monkey_states == 4

        monkey_x_positions = state.level.monkey_positions[:, 0]
        reached_right_position = monkey_x_positions >= 146
        new_monkey_states = jnp.where(
            in_state_4 & reached_right_position,
            5,
            new_monkey_states,
        )

        in_state_5 = new_monkey_states == 5

        monkey_y_positions = state.level.monkey_positions[:, 1]
        reached_top_position = monkey_y_positions <= 5

        new_monkey_states = jnp.where(
            in_state_5 & reached_top_position,
            0,
            new_monkey_states,
        )

        def update_single_monkey_position(
            state_monkey, position_monkey, new_state_monkey, step_counter
        ):
            should_update = step_counter % 16 == 0

            pos_state_0 = jnp.array([152, 5])

            pos_state_1 = jnp.where(
                state_monkey == 0,
                jnp.array([152, 5]),
                jnp.array([position_monkey[0], position_monkey[1] + 8]),
            )

            pos_state_2 = jnp.array([position_monkey[0] - 3, position_monkey[1]])

            pos_state_3 = position_monkey

            pos_state_4 = jnp.array([position_monkey[0] + 3, position_monkey[1]])

            pos_state_5 = jnp.where(
                state_monkey == 1,
                jnp.array([146, position_monkey[1]]),
                jnp.array([position_monkey[0], position_monkey[1] - 16]),
            )

            def new_state(state_monkey):
                return jnp.array(
                    [
                        (state_monkey == 0),
                        (state_monkey == 1),
                        (state_monkey == 2),
                        (state_monkey == 3),
                        (state_monkey == 4),
                        (state_monkey == 5),
                    ]
                )

            new_pos = jnp.select(
                new_state(new_state_monkey),
                [
                    pos_state_0,
                    pos_state_1,
                    pos_state_2,
                    pos_state_3,
                    pos_state_4,
                    pos_state_5,
                ],
                default=position_monkey,
            )

            return jnp.where(should_update, new_pos, position_monkey)

        new_monkey_positions = jax.vmap(
            update_single_monkey_position, in_axes=(0, 0, 0, None)
        )(
            state.level.monkey_states,
            state.level.monkey_positions,
            new_monkey_states,
            state.level.step_counter,
        )

        def update_timer(new_state, old_state, current_timer, step_counter):
            return jnp.where(
                new_state == 3,
                jnp.where(
                    old_state == 2,
                    4,
                    jnp.where(step_counter % 16 == 0, current_timer - 1, current_timer),
                ),
                current_timer,
            )

        new_monkey_throw_timers = jax.vmap(update_timer, in_axes=(0, 0, 0, None))(
            new_monkey_states,
            state.level.monkey_states,
            state.level.monkey_throw_timers,
            state.level.step_counter,
        )

        def update_coco_state(
            old_m_state, new_m_state, old_m_timer, new_m_timer, c_state, c_pos_x
        ):
            return jnp.where(
                (old_m_state != 3) & (new_m_state == 3),
                1,
                jnp.where(
                    (c_state == 1) & (old_m_timer == 3) & (new_m_timer == 2),
                    2,
                    jnp.where(c_pos_x <= 15, 0, c_state),
                ),
            )

        new_coco_states = jax.vmap(update_coco_state, in_axes=(0, 0, 0, 0, 0, 0))(
            state.level.monkey_states,
            new_monkey_states,
            state.level.monkey_throw_timers,
            new_monkey_throw_timers,
            state.level.coco_states,
            state.level.coco_positions[:, 0],
        )

        def update_coco_positions(new_c_state, old_c_state, stepc, old_c_pos, new_m_pos):
            return jnp.where(
                new_c_state == 2,
                jnp.where(
                    stepc % 2 == 0,
                    jnp.array([old_c_pos[0] - 2, old_c_pos[1]]),
                    old_c_pos,
                ),
                jnp.where(
                    (new_c_state == 1) & (old_c_state == 0),
                    jnp.array(
                        [
                            new_m_pos[0] - 6,
                            jnp.where(
                                state.level.spawn_position,
                                new_m_pos[1] - 5,
                                new_m_pos[1] + self.consts.MONKEY_HEIGHT - self.consts.COCONUT_HEIGHT,
                            ),
                        ]
                    ),
                    old_c_pos,
                ),
            )

        new_coco_positions = jax.vmap(update_coco_positions, in_axes=(0, 0, None, 0, 0))(
            new_coco_states,
            state.level.coco_states,
            state.level.step_counter,
            state.level.coco_positions,
            new_monkey_positions,
        )

        # Handle punching at the very end, after all other state transitions to avoid race conditions
        fist_w = 3
        fist_h = 4
        fist_x = jnp.where(
            state.player.orientation > 0,
            state.player.x + self.consts.PLAYER_WIDTH,
            state.player.x - fist_w,
        )
        fist_y = state.player.y + 8

        def check_punch(f_x, f_y, f_w, f_h, m_x, m_y, m_w, m_h, m_state, punching):
            return jnp.logical_and(
                self._entities_collide(f_x, f_y, f_w, f_h, m_x, m_y, m_w, m_h),
                jnp.logical_and(m_state != 0, punching),
            )

        monkeys_punched = jax.vmap(
            check_punch,
            in_axes=(None, None, None, None, 0, 0, None, None, 0, None),
        )(
            fist_x,
            fist_y,
            fist_w,
            fist_h,
            state.level.monkey_positions[:, 0],
            state.level.monkey_positions[:, 1],
            self.consts.MONKEY_WIDTH,
            self.consts.MONKEY_HEIGHT,
            state.level.monkey_states,
            punching,
        )

        score_addition = jnp.sum(monkeys_punched) * 200

        new_monkey_states = jax.vmap(lambda a, b: jnp.where(b, 0, a), in_axes=(0, 0))(
            new_monkey_states, monkeys_punched
        )
        new_monkey_positions = jax.vmap(
            lambda pos, punched: jnp.where(punched, jnp.array([152, 5]), pos),
            in_axes=(0, 0)
        )(new_monkey_positions, monkeys_punched)

        flip = jnp.any((state.level.monkey_states != 3) & (new_monkey_states == 3))

        return (
            new_monkey_states,
            new_monkey_positions,
            new_monkey_throw_timers,
            score_addition,
            new_coco_positions,
            new_coco_states,
            flip,
        )

    @partial(jax.jit, static_argnums=(0,))
    def obs_to_flat_array(self, obs: KangarooObservation) -> chex.Array:
        """Converts the observation to a flat array."""
        return jnp.concatenate(
            [
                obs.player_x.flatten(),
                obs.player_y.flatten(),
                obs.player_o.flatten(),
                obs.platform_positions.flatten(),
                obs.ladder_positions.flatten(),
                obs.fruit_positions.flatten(),
                obs.bell_position.flatten(),
                obs.child_position.flatten(),
                obs.falling_coco_position.flatten(),
                obs.coco_positions.flatten(),
                obs.monkey_positions.flatten(),
            ]
        )

    def render(self, state: KangarooState) -> jnp.ndarray:
        return self.renderer.render(state)

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Dict:
        """Returns the observation space for Kangaroo.
        The observation contains:
        - player_x: int (0-160)
        - player_y: int (0-210)
        - player_o: int (-1 or 1 for orientation)
        - platform_positions: array of shape (20, 2) with x,y coordinates (0-160, 0-210)
        - ladder_positions: array of shape (20, 2) with x,y coordinates (0-160, 0-210)
        - fruit_positions: array of shape (3, 2) with x,y coordinates (0-160, 0-210)
        - bell_position: array of shape (2,) with x,y coordinates (0-160, 0-210)
        - child_position: array of shape (2,) with x,y coordinates (0-160, 0-210)
        - falling_coco_position: array of shape (2,) with x,y coordinates (0-160, 0-210)
        - monkey_positions: array of shape (4, 2) with x,y coordinates (0-160, 0-210)
        - coco_positions: array of shape (4, 2) with x,y coordinates (0-160, 0-210)
        """
        return spaces.Dict(
            {
                "player_x": spaces.Box(low=0, high=160, shape=(), dtype=jnp.int32),
                "player_y": spaces.Box(low=0, high=210, shape=(), dtype=jnp.int32),
                "player_o": spaces.Box(low=-1, high=1, shape=(), dtype=jnp.int32),
                "platform_positions": spaces.Box(
                    low=0, high=160, shape=(20, 2), dtype=jnp.int32
                ),
                "ladder_positions": spaces.Box(
                    low=0, high=160, shape=(20, 2), dtype=jnp.int32
                ),
                "fruit_positions": spaces.Box(
                    low=0, high=160, shape=(3, 2), dtype=jnp.int32
                ),
                "bell_position": spaces.Box(
                    low=0, high=160, shape=(2,), dtype=jnp.int32
                ),
                "child_position": spaces.Box(
                    low=0, high=160, shape=(2,), dtype=jnp.int32
                ),
                "falling_coco_position": spaces.Box(
                    low=0, high=160, shape=(2,), dtype=jnp.int32
                ),
                "monkey_positions": spaces.Box(
                    low=0, high=160, shape=(4, 2), dtype=jnp.int32
                ),
                "coco_positions": spaces.Box(
                    low=0, high=160, shape=(4, 2), dtype=jnp.int32
                ),
            }
        )

    def image_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(210, 160, 3),
            dtype=jnp.uint8
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key=None) -> Tuple[
        KangarooObservation,
        KangarooState,
    ]:
        state = self.reset_level(1)
        obs = self._get_observation(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def reset_level(self, next_level=1) -> KangarooState:
        next_level = jnp.clip(next_level, 1, 3)
        level_constants: LevelConstants = self._get_level_constants(next_level)

        new_state = KangarooState(
            player=PlayerState(
                x=jnp.array(self.consts.PLAYER_START_X),
                y=jnp.array(self.consts.PLAYER_START_Y),
                vel_x=jnp.array(0),
                is_crouching=jnp.array(False),
                is_jumping=jnp.array(False),
                is_climbing=jnp.array(False),
                jump_counter=jnp.array(0),
                orientation=jnp.array(1),
                jump_base_y=jnp.array(self.consts.PLAYER_START_Y),
                landing_base_y=jnp.array(self.consts.PLAYER_START_Y),
                height=jnp.array(self.consts.PLAYER_HEIGHT),
                jump_orientation=jnp.array(0),
                climb_base_y=jnp.array(self.consts.PLAYER_START_Y),
                climb_counter=jnp.array(0),
                punch_left=jnp.array(False),
                punch_right=jnp.array(False),
                cooldown_counter=jnp.array(0),
                chrash_timer=jnp.array(0),
                is_crashing=jnp.array(False),
                last_stood_on_platform_y=jnp.array(1000),
                walk_animation=jnp.array(0),
                punch_counter=jnp.array(0),
                needs_release=jnp.array(False),
            ),
            level=LevelState(
                bell_position=level_constants.bell_position,
                bell_timer=jnp.array(0),
                fruit_positions=level_constants.fruit_positions,
                fruit_actives=jnp.ones(3, dtype=jnp.bool_),
                fruit_stages=jnp.zeros(3, dtype=jnp.int32),
                ladder_positions=level_constants.ladder_positions,
                ladder_sizes=level_constants.ladder_sizes,
                platform_positions=level_constants.platform_positions,
                platform_sizes=level_constants.platform_sizes,
                child_position=level_constants.child_position,
                child_timer=jnp.array(0),
                child_velocity=jnp.array(1),
                timer=jnp.array(2000),  # to be modified
                falling_coco_position=jnp.array([13, -1]),
                falling_coco_dropping=jnp.array(False),
                falling_coco_counter=jnp.array(0),
                falling_coco_skip_update=jnp.array(False),
                step_counter=jnp.array(0),
                monkey_states=jnp.zeros(4, dtype=jnp.int32),
                monkey_positions=jnp.array([[152, 5], [152, 5], [152, 5], [152, 5]]),
                monkey_throw_timers=jnp.zeros(4, dtype=jnp.int32),
                spawn_protection=jnp.array(True),
                coco_positions=jnp.array(
                    [[-10, -10], [-10, -10], [-10, -10], [-10, -10]]
                ),
                coco_states=jnp.zeros(4, dtype=jnp.int32),
                spawn_position=jnp.array(False),
                bell_animation=jnp.array(0),
            ),
            score=jnp.array(0),
            current_level=next_level,
            level_finished=jnp.array(False),
            levelup_timer=jnp.array(0),
            reset_coords=jnp.array(False),
            levelup=jnp.array(False),
            lives=jnp.array(3),
        )
        return new_state

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def step(self, state: KangarooState, action: chex.Array) -> Tuple[KangarooObservation, KangarooState, float, bool, KangarooInfo]:
        reset_cond = jnp.any(jnp.array([action == self.consts.RESET]))

        (
            player_x,
            player_y,
            vel_x,
            is_crouching,
            is_jumping,
            is_climbing,
            jump_counter,
            orientation,
            jump_base_y,
            landing_base_y,
            new_player_height,
            new_jump_orientation,
            climb_base_y,
            climb_counter,
            punch_left,
            punch_right,
            cooldown_counter,
            level_finished,
            punch_counter,
            needs_release,
        ) = self._player_step(state, action)

        new_current_level, new_levelup_timer, new_reset_coords, new_levelup = (
            self._next_level(state)
        )

        # Handle fruit collection
        fruit_score_addition, new_actives, new_fruit_stages, bell_timer = self._fruits_step(
            state
        )
        child_timer, new_child_x, new_child_y, new_child_velocity = self._child_step(state)

        new_main_timer = self._timer_controller(state)

        (
            new_falling_coco_position,
            new_falling_coco_dropping,
            new_falling_coco_counter,
            new_falling_coco_skip_update,
            falling_coco_score_addition,
        ) = self._falling_coconut_controller(state, punch_left | punch_right)

        (
            new_monkey_states,
            new_monkey_positions,
            new_monkey_throw_timers,
            monkey_hit_score_addition,
            new_coco_positions,
            new_coco_states,
            flip,
        ) = self._monkey_controller(state, (punch_left | punch_right))

        (
            new_lives,
            new_is_crashing,
            crash_timer,
            crash_timer_done,
            new_last_stood_on_platform_y,
        ) = self._lives_controller(state)

        # add the time after finishing a level
        level_switch_score_addition = jnp.where(level_finished, state.level.timer, 0)

        # add score if levelup from lvl3 to lvl1
        score_addition = (
            fruit_score_addition
            + monkey_hit_score_addition
            + level_switch_score_addition
            + falling_coco_score_addition
        )
        score_addition = jax.lax.cond(
            new_current_level == 4,
            lambda: score_addition + 1400,
            lambda: score_addition,
        )
        new_current_level = jnp.where(new_current_level == 4, 1, new_current_level)

        new_bell_animation_timer = jnp.where(
            bell_timer > 0,
            jnp.where(state.level.bell_animation == 0, 192, state.level.bell_animation),
            jnp.where(
                state.level.bell_animation > 0,
                state.level.bell_animation - 1,
                state.level.bell_animation,
            ),
        )

        new_level_state = jax.lax.cond(
            new_levelup,
            lambda: self.reset_level(new_current_level).level,
            lambda: jax.lax.cond(
                crash_timer_done,
                lambda: self.reset_level(state.current_level).level,
                lambda: LevelState(
                    bell_position=state.level.bell_position,
                    fruit_positions=state.level.fruit_positions,
                    ladder_positions=state.level.ladder_positions,
                    ladder_sizes=state.level.ladder_sizes,
                    platform_positions=state.level.platform_positions,
                    platform_sizes=state.level.platform_sizes,
                    child_position=jnp.array([new_child_x, new_child_y]),
                    timer=new_main_timer,
                    bell_timer=bell_timer,
                    child_timer=child_timer,
                    child_velocity=new_child_velocity,
                    fruit_actives=new_actives,
                    fruit_stages=new_fruit_stages,
                    falling_coco_position=jnp.where(
                        state.levelup_timer == 0,
                        new_falling_coco_position,
                        state.level.falling_coco_position,
                    ),
                    falling_coco_dropping=new_falling_coco_dropping,
                    falling_coco_counter=new_falling_coco_counter,
                    falling_coco_skip_update=new_falling_coco_skip_update,
                    step_counter=(state.level.step_counter + 1) % 256,
                    monkey_positions=jnp.where(
                        state.levelup_timer == 0,
                        new_monkey_positions,
                        state.level.monkey_positions,
                    ),
                    monkey_states=new_monkey_states,
                    monkey_throw_timers=new_monkey_throw_timers,
                    spawn_protection=jnp.where(
                        (state.level.step_counter == 255)
                        & state.level.spawn_protection,
                        False,
                        state.level.spawn_protection,
                    ),
                    coco_positions=new_coco_positions,
                    coco_states=new_coco_states,
                    spawn_position=jnp.where(
                        flip,
                        ~state.level.spawn_position,
                        state.level.spawn_position,
                    ),
                    bell_animation=new_bell_animation_timer,
                ),
            ),
        )

        currently_walking = jnp.logical_or(
            jnp.logical_or(
                jnp.logical_or(action == Action.RIGHT, action == Action.LEFT),
                jnp.logical_or(action == Action.UPRIGHT, action == Action.UPLEFT),
            ),
            jnp.logical_or(action == Action.DOWNRIGHT, action == Action.DOWNLEFT),
        )
        new_walk_counter = jnp.where(
            currently_walking, state.player.walk_animation + 1, 0
        )

        new_walk_counter = jnp.where(new_walk_counter == 16, 0, new_walk_counter)

        new_player_state = jax.lax.cond(
            crash_timer_done,
            lambda: self.reset_level(state.current_level).player,
            lambda: PlayerState(
                x=player_x,
                y=player_y,
                vel_x=vel_x,
                is_crouching=is_crouching,
                is_jumping=is_jumping,
                is_climbing=is_climbing,
                jump_counter=jump_counter,
                orientation=orientation,
                jump_base_y=jump_base_y,
                landing_base_y=landing_base_y,
                height=new_player_height,
                jump_orientation=new_jump_orientation,
                climb_base_y=climb_base_y,
                climb_counter=climb_counter,
                punch_left=punch_left,
                punch_right=punch_right,
                cooldown_counter=cooldown_counter,
                chrash_timer=crash_timer,
                is_crashing=new_is_crashing,
                last_stood_on_platform_y=new_last_stood_on_platform_y,
                walk_animation=new_walk_counter,
                punch_counter=punch_counter,
                needs_release=needs_release,
            ),
        )
        new_state = jax.lax.cond(
            reset_cond,
            lambda: self.reset_level(1),
            lambda: KangarooState(
                player=new_player_state,
                level=new_level_state,
                score=state.score + score_addition,
                current_level=new_current_level,
                level_finished=level_finished,
                levelup_timer=new_levelup_timer,
                reset_coords=new_reset_coords,
                levelup=new_levelup,
                lives=new_lives,
            ),
        )
        done = self._get_done(new_state)
        env_reward = self._get_env_reward(state, new_state)
        all_rewards = self._get_all_rewards(state, new_state)
        info = self._get_info(new_state, all_rewards)

        observation = self._get_observation(new_state)

        return observation, new_state, env_reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: KangarooState) -> KangarooObservation:
        fruit_mask = state.level.fruit_actives[:, jnp.newaxis]
        fruit_positions = jnp.where(
            fruit_mask, state.level.fruit_positions, jnp.array([-1, -1])
        )

        bell_mask = state.level.bell_position[jnp.newaxis, :]
        bell_position = jnp.where(
            bell_mask, state.level.bell_position, jnp.array([-1, -1])
        )

        falling_coco_mask = state.level.falling_coco_dropping[None]
        falling_coco_position = jnp.where(
            falling_coco_mask, state.level.falling_coco_position, jnp.array([-1, -1])
        )

        monkey_mask = state.level.monkey_states[:, jnp.newaxis]
        monkey_positions = jnp.where(
            monkey_mask, state.level.monkey_positions, jnp.array([-1, -1])
        )

        coco_mask = state.level.coco_states[:, jnp.newaxis]
        coco_positions = jnp.where(
            coco_mask, state.level.coco_positions, jnp.array([-1, -1])
        )

        return KangarooObservation(
            player_x=state.player.x,
            player_y=state.player.y,
            player_o=state.player.orientation,
            platform_positions=state.level.platform_positions,
            ladder_positions=state.level.ladder_positions,
            fruit_positions=fruit_positions,
            bell_position=bell_position,
            child_position=state.level.child_position,
            falling_coco_position=falling_coco_position,
            monkey_positions=monkey_positions,
            coco_positions=coco_positions,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_info(self, state: KangarooState, all_rewards: chex.Array) -> KangarooInfo:
        return KangarooInfo(
            score=state.score,
            level=state.current_level,
            all_rewards=all_rewards,
        )

    @partial(jax.jit, static_argnums=(0,))
    def _get_env_reward(
        self, previous_state: KangarooState, state: KangarooState
    ) -> float:
        return state.score - previous_state.score

    @partial(jax.jit, static_argnums=(0,))
    def _get_all_rewards(
        self, previous_state: KangarooState, state: KangarooState
    ) -> chex.Array:
        if self.reward_funcs is None:
            return jnp.zeros(1)
        rewards = jnp.array(
            [reward_func(previous_state, state) for reward_func in self.reward_funcs]
        )
        return rewards

    @partial(jax.jit, static_argnums=(0,))
    def _get_done(self, state: KangarooState) -> bool:
        return jnp.logical_and(state.lives <= 0, state.player.y == 188)


class KangarooRenderer(JAXGameRenderer):
    # Type hint for sprites dictionary
    sprites: Dict[str, Any]
    pivots: Dict[str, Any]

    def __init__(self, consts=None):
        """
        Initializes the renderer by loading sprites, including level backgrounds.

        Args:
            sprite_path: Path to the directory containing sprite .npy files.
        """
        self.consts = consts or KangarooConstants()
        self.sprite_path = f"{os.path.dirname(os.path.abspath(__file__))}/sprites/kangaroo"
        self.sprites, self.pivots = self._load_sprites()
        # Store background sprites directly for use in render function
        self.background_0 = self.sprites.get('background_0')
        self.background_1 = self.sprites.get('background_1')
        self.background_2 = self.sprites.get('background_2')
        # ladder constants
        self.ladder_rung_height = 4  # The height of the solid bar
        self.ladder_space_height = 4 # The height of the empty space
        self.ladder_color = jnp.array([162, 98, 33], dtype=jnp.uint8)
        # platform constants
        self.platform_color = jnp.array([162, 98, 33], dtype=jnp.uint8)

    def _load_sprites(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Loads all necessary sprites from .npy files and stores pivots."""
        sprites: Dict[str, Any] = {}
        pad_offsets: Dict[str, Any] = {}

        # Helper function to load a single sprite frame
        def _load_sprite_frame(name: str) -> Optional[chex.Array]:
            path = os.path.join(self.sprite_path, f'{name}.npy')
            frame = jr.loadFrame(path)
            if isinstance(frame, jnp.ndarray) and frame.ndim >= 2:
                return frame.astype(jnp.uint8)

        # --- Load Sprites ---
        # Backgrounds + Dynamic elements + UI elements
        sprite_names = [
            'background',
            'ape_climb_left', 'ape_climb_right', 'ape_moving', 'ape_standing',
            'bell', 'ringing_bell', 'child_jump', 'child', 'coconut', 'kangaroo',
            'kangaroo_climb', 'kangaroo_dead', 'kangaroo_ducking',
            'kangaroo_jump_high', 'kangaroo_jump', 'kangaroo_lives',
            'kangaroo_walk', 'kangaroo_boxing',
            'strawberry', 'tomato', 'cherry', 'pineapple', 'throwing_ape', 'thrown_coconut', 'time_dash',
        ]
        for name in sprite_names:
            loaded_sprite = _load_sprite_frame(name)
            if loaded_sprite is not None:
                sprites[name] = loaded_sprite
        # import ipdb; ipdb.set_trace()
        #s = sprites['ape_climb_left']
        #plt.imshow(s)
        #plt.show()

        # pad the kangaroo and monkey sprites since they have to be used interchangeably (and jax enforces same sizes)
        ape_sprites, ape_pivots = jr.pad_to_match([
            sprites['ape_climb_left'], 
            sprites['ape_climb_right'], 
            sprites['ape_moving'], 
            sprites['ape_standing'], 
            sprites['throwing_ape']
        ])
        ape_keys = ['ape_climb_left', 'ape_climb_right', 'ape_moving', 'ape_standing', 'throwing_ape']
        for i, key in enumerate(ape_keys):
            sprites[key] = ape_sprites[i]
            pad_offsets[key] = ape_pivots[i]

        # --- pad kangaroo ---
        kangaroo_sprites, kangaroo_pivots = jr.pad_to_match([
            sprites['kangaroo'],
            sprites['kangaroo_climb'],
            sprites['kangaroo_dead'],
            sprites['kangaroo_ducking'],
            sprites['kangaroo_jump_high'],
            sprites['kangaroo_jump'],
            sprites['kangaroo_walk'],
            sprites['kangaroo_boxing']
        ])
        kangaroo_keys = [
            'kangaroo', 'kangaroo_climb', 'kangaroo_dead', 'kangaroo_ducking',
            'kangaroo_jump_high', 'kangaroo_jump', 'kangaroo_walk', 'kangaroo_boxing'
        ]
        for i, key in enumerate(kangaroo_keys):
            sprites[key] = kangaroo_sprites[i]
            pad_offsets[key] = kangaroo_pivots[i]

        # pad bell / ringing bell
        bell_sprites, bell_pivots = jr.pad_to_match([sprites['bell'], sprites['ringing_bell']])
        bell_keys = ['bell', 'ringing_bell']
        for i, key in enumerate(bell_keys):
            sprites[key] = bell_sprites[i]
            pad_offsets[key] = bell_pivots[i]

        # pad fruits
        fruit_sprites, fruit_pivots = jr.pad_to_match([
            sprites['strawberry'], sprites['tomato'], sprites['cherry'], sprites['pineapple']
        ])
        fruit_keys = ['strawberry', 'tomato', 'cherry', 'pineapple']
        for i, key in enumerate(fruit_keys):
            sprites[key] = fruit_sprites[i]
            pad_offsets[key] = fruit_pivots[i]

        # pad child sprites
        child_sprites, child_pivots = jr.pad_to_match([
            sprites['child'], sprites['child_jump']
        ])
        child_keys = ['child', 'child_jump']
        for i, key in enumerate(child_keys):
            sprites[key] = child_sprites[i]
            pad_offsets[key] = child_pivots[i]

        # --- Load Digit Sprites ---
        # Score digits
        score_digit_path = os.path.join(self.sprite_path, 'score_{}.npy')
        digits = jr.load_and_pad_digits(score_digit_path, num_chars=10)
        sprites['digits'] = digits
        # Time digits
        time_digit_path = os.path.join(self.sprite_path, 'time_{}.npy')
        time_digits = jr.load_and_pad_digits(time_digit_path, num_chars=10)
        sprites['time_digits'] = time_digits

        # expand all sprites similar to the Pong/Seaquest loading
        for key in sprites.keys():
            if isinstance(sprites[key], (list, tuple)):
                sprites[key] = [jnp.expand_dims(sprite, axis=0) for sprite in sprites[key]]
            else:
                sprites[key] = jnp.expand_dims(sprites[key], axis=0)

        return sprites, pad_offsets

    # Apply JIT compilation. static_argnums=(0,) means 'self' is static.
    @partial(jax.jit, static_argnums=(0,))
    def render(self, state: KangarooState) -> chex.Array:
        """
        Renders the current game state to a JAX array (raster image)
        using pre-rendered backgrounds per level.

        Args:
            state: The current KangarooState.

        Returns:
            A JAX array representing the rendered game screen (HEIGHT, WIDTH, 3), dtype=uint8.
        """

        # --- Select and Render Background ---
        # Initialize raster using the consistent function
        raster = jr.create_initial_frame(width=160, height=210)

        background_sprite = self.sprites.get('background')
        background_sprite = jr.get_sprite_frame(background_sprite, 0)
        raster = jr.render_at(raster, 0, 0, background_sprite)

        # --- Draw the current platforms ---

        def create_single_platform_mask(pos, size, xx, yy):
            """Generates a boolean mask for one rectangular platform."""
            should_draw = pos[0] != -1

            def draw_fn():
                """Calculates and returns the actual platform mask."""
                x_start, y_start = pos[0], pos[1]
                width, height = size[0], size[1]
                return (xx >= x_start) & (xx < x_start + width) & \
                    (yy >= y_start) & (yy < y_start + height)

            def no_draw_fn():
                """Returns an empty mask."""
                return jnp.zeros_like(xx, dtype=bool)

            return jax.lax.cond(should_draw, draw_fn, no_draw_fn)

        platform_positions = state.level.platform_positions
        platform_sizes = state.level.platform_sizes
        platform_color = self.platform_color

        # Create coordinate grids once for the entire raster
        xx, yy = jnp.meshgrid(jnp.arange(self.consts.SCREEN_WIDTH), jnp.arange(self.consts.SCREEN_HEIGHT), indexing='xy')

        # Use vmap to create all platform masks in parallel
        vmap_platform_mask = jax.vmap(
            create_single_platform_mask,
            in_axes=(0, 0, None, None)
        )
        all_platform_masks = vmap_platform_mask(platform_positions, platform_sizes, xx, yy)

        # Reduce the stack of masks into a single one with a logical OR
        combined_platform_mask = jnp.logical_or.reduce(all_platform_masks, axis=0)

        # Apply the combined mask to the raster in one operation
        raster = jnp.where(combined_platform_mask[..., None], platform_color, raster)
    

        def create_single_ladder_mask(pos, size, xx, yy, rung_height, space_height):
            """Generates a boolean mask for one ladder using jax.lax.cond."""
            should_draw = pos[0] != -1

            def draw_fn():
                """Calculates and returns the full ladder mask."""
                x_start, y_start = pos[0], pos[1]
                width, hitbox_height = size[0], size[1]
                pattern_height = rung_height + space_height

                # Calculate number of rungs and the final visual height
                num_rungs = jnp.ceil((hitbox_height + space_height) / pattern_height).astype(int)
                visual_height = (num_rungs * rung_height) + ((num_rungs - 1) * space_height)

                # Create the mask for this ladder
                area_mask = (xx >= x_start) & (xx < x_start + width) & \
                            (yy >= y_start) & (yy < y_start + visual_height)
                relative_y = yy - y_start
                pattern_mask = (relative_y % pattern_height) < rung_height
                
                return area_mask & pattern_mask

            def no_draw_fn():
                """Returns an empty mask for invalid ladders."""
                return jnp.zeros_like(xx, dtype=bool)

            # Use lax.cond to choose which function to execute.
            return jax.lax.cond(should_draw, draw_fn, no_draw_fn)

        # --- Draw the current ladders ---
        ladder_positions = state.level.ladder_positions
        ladder_sizes = state.level.ladder_sizes

        # Use vmap to create all ladder masks in parallel
        vmap_ladder_mask = jax.vmap(
            create_single_ladder_mask,
            in_axes=(0, 0, None, None, None, None)
        )
        all_masks = vmap_ladder_mask(
            ladder_positions,
            ladder_sizes,
            xx,
            yy,
            self.ladder_rung_height,
            self.ladder_space_height
        )

        # Reduce the stack of masks into a single mask with a logical OR
        combined_mask = jnp.logical_or.reduce(all_masks, axis=0)

        # Apply the combined mask to the raster in one operation
        raster = jnp.where(combined_mask[..., None], self.ladder_color, raster)

        # --- Draw fruits ---
        fruit_positions = state.level.fruit_positions
        fruit_actives = state.level.fruit_actives

        def _draw_fruit(i, current_raster):
            # get the current fruit type using the fruit_stages array
            fruit_type = state.level.fruit_stages[i].astype(int)

            fruit_sprite = jax.lax.switch(
                fruit_type,
                [
                    lambda: self.sprites.get('strawberry'), # Case 0
                    lambda: self.sprites.get('tomato'),    # Case 1
                    lambda: self.sprites.get('cherry'),    # Case 2
                    lambda: self.sprites.get('pineapple'),  # Case 3
                ]
            )

            fruit_pivot = jax.lax.switch(
                fruit_type,
                [
                    lambda: self.pivots.get('strawberry', jnp.array([0.0, 0.0])),
                    lambda: self.pivots.get('tomato', jnp.array([0.0, 0.0])),
                    lambda: self.pivots.get('cherry', jnp.array([0.0, 0.0])),
                    lambda: self.pivots.get('pineapple', jnp.array([0.0, 0.0])),
                ]
            )

            should_draw = jnp.logical_and(fruit_actives[i], fruit_sprite is not None)
            pos = fruit_positions[i]
            def render_fruit_sprite(raster_to_update):
                return jr.render_at(raster_to_update, pos[0].astype(int), pos[1].astype(int), jr.get_sprite_frame(fruit_sprite, 0), flip_offset=fruit_pivot)
            return jax.lax.cond(should_draw, render_fruit_sprite, lambda r: r, current_raster)

        num_fruits_to_draw = fruit_positions.shape[0]
        raster = jax.lax.fori_loop(0, num_fruits_to_draw, _draw_fruit, raster)

        # --- Draw Bell ---
        # if the bell_animation is: 192-176, 143-128, 95-80, 47-32 draw the alternate bell sprite
        bell_in_range_left = jnp.logical_or(
            jnp.logical_and(state.level.bell_animation <= 192, state.level.bell_animation >= 176),
            jnp.logical_and(state.level.bell_animation <= 95, state.level.bell_animation >= 80),
        )

        bell_in_range_right = jnp.logical_or(
            jnp.logical_and(state.level.bell_animation <= 143, state.level.bell_animation >= 128),
            jnp.logical_and(state.level.bell_animation <= 47, state.level.bell_animation >= 32)
        )

        bell_sprite = jax.lax.cond(
            jnp.logical_or(bell_in_range_left, bell_in_range_right),
            lambda: self.sprites.get('ringing_bell'),
            lambda: self.sprites.get('bell')
        )
        bell_pivot = jax.lax.cond(
            jnp.logical_or(bell_in_range_left, bell_in_range_right),
            lambda: self.pivots.get('ringing_bell', jnp.array([0.0, 0.0])),
            lambda: self.pivots.get('bell', jnp.array([0.0, 0.0]))
        )

        bell_pos = state.level.bell_position
        not_all_fruits_collected = ~jnp.any(state.level.fruit_stages == 3)
        bell_pos_valid = bell_pos[0] != -1
        sprite_is_valid = bell_sprite is not None
        should_draw_bell = jnp.logical_and(jnp.logical_and(not_all_fruits_collected, bell_pos_valid), sprite_is_valid)

        def draw_bell_func(current_raster):
            return jr.render_at(current_raster, bell_pos[0].astype(int), bell_pos[1].astype(int), jr.get_sprite_frame(bell_sprite, 0), flip_horizontal=bell_in_range_left, flip_offset=bell_pivot)
        raster = jax.lax.cond(should_draw_bell, draw_bell_func, lambda r: r, raster)

        # --- Draw monkeys (Apes) ---
        monkey_positions = state.level.monkey_positions
        monkey_states = state.level.monkey_states

        def _draw_monkey(i, current_raster):
            state_idx = monkey_states[i].astype(int)
            pos = monkey_positions[i]
            should_draw = state_idx != 0
            """
            - 0: non-existent
            - 1: moving down
            - 2: moving left
            - 3: throwing
            - 4: moving right
            - 5: moving up
            """
            monkey_sprite = jax.lax.switch(
                state_idx,
                [
                    lambda: self.sprites.get('ape_standing'), # Case 0
                    lambda: self.sprites.get('ape_climb_left'),    # Case 1
                    lambda: self.sprites.get('ape_moving'),    # Case 2
                    lambda: self.sprites.get('throwing_ape'),  # Case 3
                    lambda: self.sprites.get('ape_moving'),    # Case 4
                    lambda: self.sprites.get('ape_climb_right'),# Case 5
                ]
            )

            monkey_pivot = jax.lax.switch(
                state_idx,
                [
                    lambda: self.pivots.get('ape_standing', jnp.array([0.0, 0.0])),
                    lambda: self.pivots.get('ape_climb_left', jnp.array([0.0, 0.0])),
                    lambda: self.pivots.get('ape_moving', jnp.array([0.0, 0.0])),
                    lambda: self.pivots.get('throwing_ape', jnp.array([0.0, 0.0])),
                    lambda: self.pivots.get('ape_moving', jnp.array([0.0, 0.0])),
                    lambda: self.pivots.get('ape_climb_right', jnp.array([0.0, 0.0])),
                ]
            )

            # in case its state_idx 2 or 4 and the counter is % 16, use standing instead of moving
            monkey_sprite, monkey_pivot = jax.lax.cond(
                jnp.logical_and(
                    (state.level.step_counter % 32) < 16,
                    jnp.logical_or(state_idx == 2, state_idx == 4)
                ),
                lambda: (self.sprites.get('ape_standing'), self.pivots.get('ape_standing', jnp.array([0.0, 0.0]))),
                lambda: (monkey_sprite, monkey_pivot)
            )

            is_moving_left = (state_idx == 4)
            flip_h = is_moving_left
            sprite_is_valid = monkey_sprite is not None
            should_draw = jnp.logical_and(should_draw, sprite_is_valid)
            def render_monkey_sprite(raster_to_update):
                return jr.render_at(raster_to_update, pos[0].astype(int), pos[1].astype(int), jr.get_sprite_frame(monkey_sprite, 0), flip_horizontal=flip_h, flip_offset=monkey_pivot)
            return jax.lax.cond(should_draw, render_monkey_sprite, lambda r: r, current_raster)

        num_monkeys_to_draw = monkey_positions.shape[0]
        raster = jax.lax.fori_loop(0, num_monkeys_to_draw, _draw_monkey, raster)

        # --- Draw player (Kangaroo) ---
        player_pos_x = state.player.x
        player_pos_y = state.player.y
        player_orientation = state.player.orientation
        flip_player = player_orientation < 0
        sprite_lambda = jax.lax.cond(
            state.player.is_crashing, lambda: self.sprites.get('kangaroo_dead'),
            lambda: jax.lax.cond(
                state.player.is_climbing, lambda: self.sprites.get('kangaroo_climb'),
                lambda: jax.lax.cond(
                    state.player.is_crouching, lambda: self.sprites.get('kangaroo_ducking'),
                    lambda: jax.lax.cond(
                        state.player.is_jumping, lambda: self.sprites.get('kangaroo_jump'),
                        lambda: jax.lax.cond(
                            state.player.punch_left | state.player.punch_right,
                            lambda: self.sprites.get('kangaroo_boxing'),
                            lambda: self.sprites.get('kangaroo')
                        )
                    )
                )
            )
        )

        # check if player.walk_animation is between 6 and 16 in which range the kangaroo has a different animation
        player_walking_animation = jnp.logical_and(state.player.walk_animation > 6, state.player.walk_animation < 16)

        # Only apply walking animation if not crouching, jumping, climbing, or crashing
        is_walking_anim = (
            player_walking_animation
            & ~state.player.is_crouching
            & ~state.player.is_jumping
            & ~state.player.is_climbing
            & ~state.player.is_crashing
        )

        # Only offset y by 1 if we are really walking (not crouching)
        player_pos_y = jnp.where(
            is_walking_anim,
            player_pos_y - 1,
            player_pos_y
        )

        sprite_lambda = jax.lax.cond(
            is_walking_anim,
            lambda: self.sprites.get('kangaroo_walk'),
            lambda: sprite_lambda
        )

        # in case the player_animation is between 17 and 25, use high jump
        sprite_lambda = jax.lax.cond(
            jnp.logical_and(state.player.jump_counter > 16, state.player.jump_counter < 25),
            lambda: self.sprites.get('kangaroo_jump_high'),
            lambda: sprite_lambda
        )

        player_sprite = sprite_lambda
        player_pivot = jax.lax.switch(
            jnp.argmax(jnp.array([
                state.player.is_crashing,
                state.player.is_climbing,
                state.player.is_crouching,
                state.player.is_jumping,
                state.player.punch_left | state.player.punch_right,
                is_walking_anim,
                True # fallback to normal kangaroo
            ])),
            [
                lambda: self.pivots.get('kangaroo_dead', jnp.array([0.0, 0.0])),
                lambda: self.pivots.get('kangaroo_climb', jnp.array([0.0, 0.0])),
                lambda: self.pivots.get('kangaroo_ducking', jnp.array([0.0, 0.0])),
                lambda: self.pivots.get('kangaroo_jump', jnp.array([0.0, 0.0])),
                lambda: self.pivots.get('kangaroo_boxing', jnp.array([0.0, 0.0])),
                lambda: self.pivots.get('kangaroo_walk', jnp.array([0.0, 0.0])),
                lambda: self.pivots.get('kangaroo', jnp.array([0.0, 0.0]))
            ]
        )
        sprite_is_valid = player_sprite is not None
        def render_player_sprite(raster_to_update):
             return jr.render_at(raster_to_update,
                                 player_pos_x.astype(int),
                                 player_pos_y.astype(int),
                                 jr.get_sprite_frame(player_sprite, 0),
                                 flip_horizontal=flip_player,
                                 flip_offset=player_pivot)
        raster = jax.lax.cond(sprite_is_valid, render_player_sprite, lambda r: r, raster)

        # --- Draw Child ---
        child_pos = state.level.child_position
        is_jumping = (state.level.step_counter % 32) < 16
        # if the velocity is negative, flip horizontal
        child_flip = state.level.child_velocity > 0
        child_sprite_lambda = jax.lax.cond(
            is_jumping, lambda: self.sprites.get('child_jump'), lambda: self.sprites.get('child')
        )
        child_sprite = child_sprite_lambda
        should_draw_child = jnp.logical_and(child_pos[0] != -1, child_sprite is not None)
        child_pivot = self.pivots.get('child', jnp.array([0.0, 0.0]))
        def draw_child_func(current_raster):
            return jr.render_at(current_raster, child_pos[0].astype(int), child_pos[1].astype(int), jr.get_sprite_frame(child_sprite, 0), flip_horizontal=child_flip, flip_offset=child_pivot)
        raster = jax.lax.cond(should_draw_child, draw_child_func, lambda r: r, raster)

        # --- Draw falling coconut ---
        falling_coco_pos = state.level.falling_coco_position
        coco_sprite = self.sprites.get('thrown_coconut', None)
        should_draw_falling_coco = jnp.logical_and(falling_coco_pos[1] != -1, coco_sprite is not None)
        coco_pivot = self.pivots.get('thrown_coconut', jnp.array([0.0, 0.0]))
        def draw_falling_coco_func(current_raster):
            return jr.render_at(current_raster, falling_coco_pos[0].astype(int), falling_coco_pos[1].astype(int), jr.get_sprite_frame(coco_sprite, 0), flip_offset=coco_pivot)
        raster = jax.lax.cond(should_draw_falling_coco, draw_falling_coco_func, lambda r: r, raster)

        # --- Draw thrown coconuts ---
        coco_positions = state.level.coco_positions
        coco_states = state.level.coco_states
        coco_sprite = self.sprites.get('coconut', None)
        coco_pivot = self.pivots.get('coconut', jnp.array([0.0, 0.0]))
        def _draw_coco(i, current_raster):
            should_draw = jnp.logical_and(coco_states[i] != 0, coco_sprite is not None)
            pos = coco_positions[i]
            def render_coco_sprite(raster_to_update):
                return jr.render_at(raster_to_update, pos[0].astype(int), pos[1].astype(int), jr.get_sprite_frame(coco_sprite, 0), flip_offset=coco_pivot)
            return jax.lax.cond(should_draw, render_coco_sprite, lambda r: r, current_raster)
        num_cocos_to_draw = coco_positions.shape[0]
        raster = jax.lax.fori_loop(0, num_cocos_to_draw, _draw_coco, raster)

        # --- Draw UI ---
        # Score
        digit_sprites = self.sprites.get('digits', None)
        score_digits_indices = jr.int_to_digits(state.score, max_digits=6)
        raster = jr.render_label(raster, 105, 182, score_digits_indices, digit_sprites[0], spacing=8)

        # Lives
        life_sprite = self.sprites.get('kangaroo_lives', None)
        lives_count = jnp.maximum(state.lives.astype(int) - 1, 0)
        raster = jr.render_indicator(raster, 15, 182, lives_count, life_sprite[0], spacing=8)

        # Timer
        time_digit_sprites = self.sprites.get('time_digits', None)
        timer_val = jnp.maximum(state.level.timer.astype(int), 0)
        timer_digits_indices = jr.int_to_digits(timer_val, max_digits=4)
        raster = jr.render_label(raster, 80, 190, timer_digits_indices, time_digit_sprites[0], spacing=4)

        # Ensure the final raster has the correct dtype
        return raster.astype(jnp.uint8)
