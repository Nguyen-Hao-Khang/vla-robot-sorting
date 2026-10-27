# ruff: noqa: E501  # Scoped imported native evidence source; no semantic changes.
"""Pure unittest: source-verified surface geometry; never object authorization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import numpy as np
from PIL import Image

from vla_robot_sorting.perception.native_multiview_surface_geometry import (
    SurfaceEvidenceError,
    evaluate_native_color_surfaces,
)


def _sha(b): return hashlib.sha256(b).hexdigest()


class NativeMultiviewSurfaceTests(TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for cam in ('head_camera', 'wrist_camera'):
            self._make(cam)

    def _make(self, camera):
        rgb = np.full((90, 120, 3), 150, dtype=np.uint8)
        rgb[10:31, 10:42] = (238, 75, 12)
        rgb[45:75, 65:103] = (0, 0, 230)
        depth = np.full((90, 120), 500., dtype=np.float64)
        self._write(camera, rgb, depth)

    def _write(self, camera, rgb=None, depth=None):
        prefix = self.root / f'{camera}_frame_000000'
        if rgb is not None:
            Image.fromarray(rgb).save(str(prefix) + '_rgb.png')
        if depth is not None:
            np.save(str(prefix) + '_depth_mm.npy', depth)
        mpath = Path(str(prefix) + '_depth-manifest.json')
        old = json.loads(mpath.read_text()) if mpath.exists() else {}
        d = {
            'schema': 'damh.robotwin-native-rgbd-capture.v1',
            'status': 'capture_written', 'source_classification': 'recorded_sim_observation',
            'camera_id': camera, 'frame_index': 0,
            'frame_index_verified_by_caller': True,
            'frame_association_source': 'probe.get_obs single-frame camera refresh',
            'timestamp_semantics': 'probe wall-clock time; not a simulator sensor timestamp',
            'capture_timestamp_utc': '2026-10-13T15:44:04.871293+00:00',
            'scene_revision': None, 'controller_eligible': False,
            'depth_encoding': 'NumPy .npy float64', 'depth_units': 'millimetres',
            'depth_convention': 'sapien3_position_camera_z_mm_masked_by_float_alpha_and_position_zbuffer_v2',
            'native_capture_executed': True, 'image_size_px': [120,90],
            'intrinsic_cv': [[100., 0., 60.], [0., 100., 45.], [0., 0., 1.]],
            'camera_to_world_cv': [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],
            'extrinsic_cv_world_to_camera_cv': [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]],
        }
        d.update(old)
        d['rgb_sha256'] = _sha(Path(str(prefix) + '_rgb.png').read_bytes())
        d['depth_sha256'] = _sha(Path(str(prefix) + '_depth_mm.npy').read_bytes())
        mpath.write_text(json.dumps(d))

    def _edit(self, camera, **kw):
        path = self.root / f'{camera}_frame_000000_depth-manifest.json'
        d = json.loads(path.read_text())
        d.update(kw)
        path.write_text(json.dumps(d))

    def test_cameras_two_metric_surface_patches(self):
        r=evaluate_native_color_surfaces(self.root)
        self.assertEqual(r['camera_count'],2)
        for cam in r['cameras']:
            self.assertEqual(len(cam['surfaces']),2)
            self.assertFalse(cam['scene_snapshot_created'])
            for s in cam['surfaces']:
                self.assertIsNone(s['object_id'])
                self.assertIsNone(s['object_type'])
                self.assertIsNone(s['object_pose_6dof'])
                self.assertIsNone(s['object_identity_confidence'])
                self.assertFalse(s['controller_eligible'])
        self.assertEqual(len(r['same_color_diagnostics']),2)
        self.assertFalse(r['cross_camera_object_correspondence_verified'])
        self.assertFalse(r['semantic_grounding_authorized'])
        self.assertFalse(r['grasp_planning_authorized'])

    def test_plane_surface_visible_not_object_dimension(self):
        r=evaluate_native_color_surfaces(self.root)
        s=next(s for s in r['cameras'][0]['surfaces'] if s['kind']=='blue_surface')
        self.assertTrue(s['visible_surface_near_horizontal_planarity_heuristic'])
        self.assertAlmostEqual(s['visible_surface_centroid_world_m'][2], .5)
        self.assertIsNone(s['object_dimensions_m'])

    def test_same_color_is_not_confirmed_correspondence(self):
        r=evaluate_native_color_surfaces(self.root)
        for row in r['same_color_diagnostics']:
            self.assertEqual(row['unpaired_color_centroid_distance_m'], 0)
            self.assertFalse(row['same_physical_surface_correspondence_verified'])
            self.assertFalse(row['sensor_timestamp_synchronized'])

    def test_rgb_tamper_rejected(self):
        p=self.root/'head_camera_frame_000000_rgb.png'
        p.write_bytes(p.read_bytes()+b'x')
        with self.assertRaisesRegex(SurfaceEvidenceError,'rgb digest mismatch'):
            evaluate_native_color_surfaces(self.root)

    def test_depth_tamper_rejected(self):
        p=self.root/'head_camera_frame_000000_depth_mm.npy'
        p.write_bytes(p.read_bytes()+b'x')
        with self.assertRaisesRegex(SurfaceEvidenceError,'depth digest mismatch'):
            evaluate_native_color_surfaces(self.root)

    def test_wrong_camera_rejected(self):
        self._edit('head_camera',camera_id='wrist_camera')
        with self.assertRaisesRegex(SurfaceEvidenceError,'camera/frame conflict'):
            evaluate_native_color_surfaces(self.root)

    def test_unverified_frame_association_rejected(self):
        self._edit('head_camera',frame_association_source='historical HDF5')
        with self.assertRaisesRegex(SurfaceEvidenceError,'same-refresh'):
            evaluate_native_color_surfaces(self.root)

    def test_source_controller_claim_rejected(self):
        self._edit('wrist_camera',controller_eligible=True)
        with self.assertRaisesRegex(SurfaceEvidenceError,'control eligibility'):
            evaluate_native_color_surfaces(self.root)

    def test_unverified_timestamp_semantics_rejected(self):
        self._edit('head_camera', timestamp_semantics='perfectly synchronized')
        with self.assertRaisesRegex(SurfaceEvidenceError,'timestamp semantics'):
            evaluate_native_color_surfaces(self.root)

    def test_scene_revision_claim_rejected(self):
        self._edit('head_camera',scene_revision='trusted_scene')
        with self.assertRaisesRegex(SurfaceEvidenceError,'scene identity'):
            evaluate_native_color_surfaces(self.root)

    def test_calibration_corruption_rejected(self):
        t=np.diag([1.,1.,-1.,1.]).tolist()
        self._edit('head_camera',camera_to_world_cv=t)
        with self.assertRaisesRegex(SurfaceEvidenceError,'rotation not rigid'):
            evaluate_native_color_surfaces(self.root)

    def test_extrinsic_mismatch_rejected(self):
        self._edit('head_camera',extrinsic_cv_world_to_camera_cv=[[1,0,0,0],[0,1,0,1],[0,0,1,0]])
        with self.assertRaisesRegex(SurfaceEvidenceError,'calibration disagree'):
            evaluate_native_color_surfaces(self.root)

    def test_no_depth_does_not_fabricate_geometry(self):
        rgb=np.full((90,120,3),150,dtype=np.uint8)
        rgb[10:31,10:42]=(238,75,12)
        depth=np.zeros((90,120),dtype=np.float64)
        self._write('head_camera',rgb,depth)
        r=evaluate_native_color_surfaces(self.root)
        self.assertEqual(r['cameras'][0]['surfaces'],[])
        self.assertFalse(r['scene_snapshot_created'])

    def test_no_colored_region_returns_empty(self):
        rgb=np.full((90,120,3),150,dtype=np.uint8)
        depth=np.full((90,120),500.,dtype=np.float64)
        self._write('wrist_camera',rgb,depth)
        r=evaluate_native_color_surfaces(self.root)
        self.assertEqual(r['cameras'][1]['surfaces'],[])
        self.assertFalse(r['controller_eligible'])

    def test_noisy_and_multilayer_depth_must_not_fake_patch(self):
        rgb=np.zeros((90,120,3),dtype=np.uint8)
        rgb[10:50,10:50]=(238,75,12)
        depth=np.full((90,120),500.,dtype=np.float64)
        depth[10:30,10:50]=100.
        depth[30:50,10:50]=800.
        self._write('head_camera',rgb,depth)
        r=evaluate_native_color_surfaces(self.root)
        self.assertEqual(r['cameras'][0]['surfaces'],[])

    def test_all_camera_pair_different_wallclock_still_not_synced(self):
        self._edit('wrist_camera',capture_timestamp_utc='2026-10-13T15:44:05.123456+00:00')
        r=evaluate_native_color_surfaces(self.root)
        self.assertAlmostEqual(r['cross_camera_wallclock_capture_delta_s'],.252163)
        self.assertFalse(r['cross_camera_sensor_timestamp_synchronized'])

    def test_shape_mismatch_rejected(self):
        self._edit('wrist_camera',image_size_px=[121,90])
        with self.assertRaisesRegex(SurfaceEvidenceError,'RGB frame shape'):
            evaluate_native_color_surfaces(self.root)

    def test_missing_frame_rejected(self):
        self._edit('head_camera', frame_index=1)
        with self.assertRaisesRegex(SurfaceEvidenceError,'camera/frame conflict'):
            evaluate_native_color_surfaces(self.root)


if __name__=='__main__':
    main(verbosity=2)
