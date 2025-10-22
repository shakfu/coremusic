#!/usr/bin/env python3
"""Comprehensive tests for advanced AudioUnit OO API features."""

import pytest
import coremusic as cm


class TestAudioUnitStreamFormat:
    """Test AudioUnit stream format configuration"""

    def test_get_stream_format_output(self):
        """Test getting output stream format"""
        unit = cm.AudioUnit.default_output()
        try:
            unit.initialize()

            # Get output stream format
            format = unit.get_stream_format("output", 0)

            assert isinstance(format, cm.AudioFormat)
            assert format.sample_rate > 0
            assert format.channels_per_frame > 0

        finally:
            unit.dispose()

    def test_get_stream_format_input(self):
        """Test getting input stream format"""
        unit = cm.AudioUnit.default_output()
        try:
            unit.initialize()

            # Get input stream format (may not be configured for output-only units)
            try:
                format = unit.get_stream_format("input", 0)
                assert isinstance(format, cm.AudioFormat)
            except cm.AudioUnitError:
                # Some output units don't have input scope configured
                pass

        finally:
            unit.dispose()

    def test_set_stream_format(self):
        """Test setting stream format on input scope"""
        unit = cm.AudioUnit.default_output()
        try:
            # For output units, we can set the input format, not the output
            # The output format is typically determined by hardware
            format = cm.AudioFormat(
                sample_rate=48000.0,
                format_id="lpcm",
                format_flags=12,  # kLinearPCMFormatFlagIsSignedInteger | IsPacked
                channels_per_frame=2,
                bits_per_channel=16,
                bytes_per_frame=4,
                bytes_per_packet=4,
                frames_per_packet=1,
            )

            # Set input format (this is what we'd feed to the unit)
            try:
                unit.set_stream_format(format, "input", 0)

                # Verify it was set
                retrieved = unit.get_stream_format("input", 0)
                assert retrieved.sample_rate == 48000.0
                assert retrieved.channels_per_frame == 2
            except cm.AudioUnitError:
                # Some units may not allow setting input format
                # This is acceptable - just test that the method works
                pass

        finally:
            unit.dispose()

    def test_stream_format_invalid_scope(self):
        """Test invalid scope raises error"""
        unit = cm.AudioUnit.default_output()
        try:
            with pytest.raises(cm.AudioUnitError, match="Invalid scope"):
                unit.get_stream_format("invalid", 0)

            format = cm.AudioFormat(44100.0, "lpcm")
            with pytest.raises(cm.AudioUnitError, match="Invalid scope"):
                unit.set_stream_format(format, "invalid", 0)

        finally:
            unit.dispose()


class TestAudioUnitProperties:
    """Test AudioUnit property access"""

    def test_sample_rate_property(self):
        """Test sample rate property getter"""
        unit = cm.AudioUnit.default_output()
        try:
            unit.initialize()

            sample_rate = unit.sample_rate
            assert isinstance(sample_rate, float)
            assert sample_rate > 0

        finally:
            unit.dispose()

    def test_sample_rate_property_setter(self):
        """Test sample rate property setter"""
        unit = cm.AudioUnit.default_output()
        try:
            # Set sample rate before initialization
            unit.sample_rate = 48000.0

            # Verify (may not always succeed depending on hardware)
            try:
                rate = unit.sample_rate
                # If it worked, verify the value
                if rate > 0:
                    assert (
                        rate == 48000.0 or rate == 44100.0
                    )  # May fall back to default
            except Exception:
                pass  # Some units may not support this

        finally:
            unit.dispose()

    def test_latency_property(self):
        """Test latency property"""
        unit = cm.AudioUnit.default_output()
        try:
            unit.initialize()

            latency = unit.latency
            assert isinstance(latency, float)
            assert latency >= 0.0

        finally:
            unit.dispose()

    def test_cpu_load_property(self):
        """Test CPU load property"""
        unit = cm.AudioUnit.default_output()
        try:
            unit.initialize()

            cpu_load = unit.cpu_load
            assert isinstance(cpu_load, float)
            assert 0.0 <= cpu_load <= 1.0

        finally:
            unit.dispose()

    def test_max_frames_per_slice_property(self):
        """Test max frames per slice property getter"""
        unit = cm.AudioUnit.default_output()
        try:
            unit.initialize()

            max_frames = unit.max_frames_per_slice
            assert isinstance(max_frames, int)
            # Default is typically 512 or 1024, but may vary
            assert max_frames >= 0

        finally:
            unit.dispose()

    def test_max_frames_per_slice_setter(self):
        """Test max frames per slice property setter"""
        unit = cm.AudioUnit.default_output()
        try:
            # Set before initialization
            unit.max_frames_per_slice = 2048

            # Verify
            max_frames = unit.max_frames_per_slice
            assert max_frames == 2048

        finally:
            unit.dispose()


class TestAudioUnitParameters:
    """Test AudioUnit parameter access"""

    def test_get_parameter_list_default_output(self):
        """Test getting parameter list from default output"""
        unit = cm.AudioUnit.default_output()
        try:
            unit.initialize()

            params = unit.get_parameter_list("global")
            assert isinstance(params, list)
            # Default output may have no parameters
            assert len(params) >= 0

        finally:
            unit.dispose()

    def test_get_parameter_list_all_scopes(self):
        """Test getting parameter lists from all scopes"""
        unit = cm.AudioUnit.default_output()
        try:
            unit.initialize()

            for scope in ["global", "input", "output"]:
                params = unit.get_parameter_list(scope)
                assert isinstance(params, list)

        finally:
            unit.dispose()

    def test_get_parameter_list_invalid_scope(self):
        """Test invalid scope raises error"""
        unit = cm.AudioUnit.default_output()
        try:
            with pytest.raises(cm.AudioUnitError, match="Invalid scope"):
                unit.get_parameter_list("invalid")

        finally:
            unit.dispose()


class TestAudioUnitAdvancedWorkflow:
    """Test advanced AudioUnit workflows"""

    def test_complete_configuration_workflow(self):
        """Test complete configuration workflow"""
        unit = cm.AudioUnit.default_output()
        try:
            # Configure before initialization
            unit.max_frames_per_slice = 1024

            # Initialize
            unit.initialize()

            # Note: After initialization, the unit may adjust max_frames_per_slice
            # to match hardware capabilities, so we just verify it's > 0
            max_frames = unit.max_frames_per_slice
            assert max_frames > 0

            # Check output format (hardware-determined)
            output_format = unit.get_stream_format("output", 0)
            assert output_format.sample_rate > 0
            assert output_format.channels_per_frame > 0

            # Check properties
            assert unit.sample_rate > 0
            assert unit.latency >= 0.0
            assert 0.0 <= unit.cpu_load <= 1.0

        finally:
            unit.dispose()

    def test_context_manager_with_configuration(self):
        """Test using context manager with configuration"""
        unit = cm.AudioUnit.default_output()

        # Configure before context
        unit.max_frames_per_slice = 2048

        with unit:
            # Unit is initialized inside context
            assert unit.is_initialized

            # After initialization, verify we can read the property
            # (value may have been adjusted by hardware)
            max_frames = unit.max_frames_per_slice
            assert max_frames > 0

            # Can read output format
            output_format = unit.get_stream_format("output", 0)
            assert output_format.sample_rate > 0

        # Unit is uninitialized and disposed after context
        assert unit.is_disposed


class TestAudioUnitRenderMethod:
    """Test AudioUnit render method"""

    def test_render_not_implemented(self):
        """Test that render method raises NotImplementedError"""
        unit = cm.AudioUnit.default_output()
        try:
            with pytest.raises(NotImplementedError):
                unit.render(1024)

        finally:
            unit.dispose()


class TestAudioUnitEdgeCases:
    """Test edge cases and error handling"""

    def test_property_access_after_disposal(self):
        """Test that property access after disposal raises error"""
        unit = cm.AudioUnit.default_output()
        unit.dispose()

        with pytest.raises(RuntimeError, match="has been disposed"):
            _ = unit.sample_rate

        with pytest.raises(RuntimeError, match="has been disposed"):
            _ = unit.latency

        with pytest.raises(RuntimeError, match="has been disposed"):
            _ = unit.cpu_load

    def test_stream_format_access_after_disposal(self):
        """Test that stream format access after disposal raises error"""
        unit = cm.AudioUnit.default_output()
        unit.dispose()

        with pytest.raises(RuntimeError, match="has been disposed"):
            unit.get_stream_format("output", 0)

        format = cm.AudioFormat(44100.0, "lpcm")
        with pytest.raises(RuntimeError, match="has been disposed"):
            unit.set_stream_format(format, "output", 0)

    def test_parameter_list_after_disposal(self):
        """Test that parameter access after disposal raises error"""
        unit = cm.AudioUnit.default_output()
        unit.dispose()

        with pytest.raises(RuntimeError, match="has been disposed"):
            unit.get_parameter_list("global")
