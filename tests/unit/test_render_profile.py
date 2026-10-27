from vla_robot_sorting.simulation.render_profile import (
    FAST_PREVIEW_RENDER_PROFILE,
    apply_fast_preview_render_profile,
)


def test_fast_preview_profile_matches_robotwin_upstream_policy() -> None:
    args: dict[str, object] = {
        "render_spp": 32,
        "render_path_depth": 8,
        "render_denoiser": "oidn",
        "other": "preserved",
    }

    apply_fast_preview_render_profile(args)

    assert args["render_spp"] == 1
    assert args["render_path_depth"] == 2
    assert args["render_denoiser"] == "none"
    assert args["other"] == "preserved"
    assert FAST_PREVIEW_RENDER_PROFILE == {
        "render_spp": 1,
        "render_path_depth": 2,
        "render_denoiser": "none",
    }
