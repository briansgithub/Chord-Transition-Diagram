"""
Beat Detection Visualization Test

This script tests the beat detection algorithm by displaying vertical bars
that illuminate when beats and subdivisions occur. The number of bars equals
the number of subdivisions per measure.

Usage:
    python beat_detection_test.py <midi_file_path>
"""

import os
import sys
import time
import threading
from typing import List, Tuple

try:
    from music21 import stream, converter, meter
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing required library: {e}", file=sys.stderr)
    print("Install with: pip install music21 matplotlib", file=sys.stderr)
    sys.exit(1)


def detect_time_signatures(midi_stream: stream.Stream) -> List[Tuple[float, meter.TimeSignature]]:
    """
    Detects all time signatures and their positions in the MIDI stream.
    Checks multiple locations: flattened stream, parts, and measures.
    
    Args:
        midi_stream: music21 Stream object
        
    Returns:
        List of tuples (offset, TimeSignature) sorted by offset
    """
    time_signatures = []
    seen_offsets = set()
    
    # Method 1: Get from flattened stream
    flattened = midi_stream.flatten()
    ts_objects = flattened.getElementsByClass(meter.TimeSignature)
    for ts in ts_objects:
        offset = float(ts.offset)
        if offset not in seen_offsets:
            time_signatures.append((offset, ts))
            seen_offsets.add(offset)
    
    # Method 2: Get from parts
    try:
        for part in midi_stream.parts:
            ts_objects = part.getElementsByClass(meter.TimeSignature)
            for ts in ts_objects:
                offset = float(ts.offset)
                if offset not in seen_offsets:
                    time_signatures.append((offset, ts))
                    seen_offsets.add(offset)
    except:
        pass
    
    # Method 3: Get from measures
    try:
        for part in midi_stream.parts:
            measures = part.getElementsByClass('Measure')
            for measure in measures:
                ts_objects = measure.getElementsByClass(meter.TimeSignature)
                for ts in ts_objects:
                    # Calculate absolute offset
                    measure_offset = float(measure.offset) if hasattr(measure, 'offset') else 0.0
                    ts_offset = float(ts.offset) if hasattr(ts, 'offset') else 0.0
                    offset = measure_offset + ts_offset
                    if offset not in seen_offsets:
                        time_signatures.append((offset, ts))
                        seen_offsets.add(offset)
    except:
        pass
    
    # Method 4: Try to get from metadata or analyze the stream
    if not time_signatures:
        # Try to get time signature from the stream's metadata
        try:
            # Check if there's a time signature in the first measure
            first_measure = None
            for part in midi_stream.parts:
                measures = part.getElementsByClass('Measure')
                if measures:
                    first_measure = measures[0]
                    break
            
            if first_measure:
                ts_objects = first_measure.getElementsByClass(meter.TimeSignature)
                if ts_objects:
                    ts = ts_objects[0]
                    time_signatures.append((0.0, ts))
        except:
            pass
    
    # If still no time signature found, try to infer from the music
    if not time_signatures:
        try:
            # Analyze the stream to get time signature
            # music21 can sometimes infer time signature from note durations
            analyzed_ts = midi_stream.analyze('timeSignature')
            if analyzed_ts:
                time_signatures.append((0.0, analyzed_ts))
                print(f"Inferred time signature from music analysis: {analyzed_ts.numerator}/{analyzed_ts.denominator}")
        except:
            pass
    
    # Method 5: Infer time signature from measure structure
    # Check if consecutive measures suggest a different grouping
    try:
        measures = []
        for part in midi_stream.parts:
            part_measures = part.getElementsByClass('Measure')
            if part_measures:
                measures = part_measures
                break
        
        if measures and len(measures) > 2:
            # Skip the first measure (might be pickup/anacrusis)
            measure_durations = [m.duration.quarterLength for m in measures[1:]]
            
            # Check if pairs of measures add up to 4.0 (suggesting 4/4 mis-grouped as 2/4)
            if len(measure_durations) >= 2:
                pair_sums = []
                for i in range(0, len(measure_durations) - 1, 2):
                    if i + 1 < len(measure_durations):
                        pair_sum = measure_durations[i] + measure_durations[i + 1]
                        pair_sums.append(pair_sum)
                
                # If most pairs sum to 4.0, it's likely 4/4
                if pair_sums and len([s for s in pair_sums if abs(s - 4.0) < 0.1]) / len(pair_sums) > 0.7:
                    inferred_4_4 = meter.TimeSignature('4/4')
                    # Check if we already have a 4/4 time signature
                    has_4_4 = any(ts.numerator == 4 and ts.denominator == 4 for _, ts in time_signatures)
                    if not has_4_4:
                        # Add inferred 4/4 at the start of the main section (after first measure)
                        if measures[1].offset > 0:
                            inferred_offset = float(measures[1].offset)
                            time_signatures.append((inferred_offset, inferred_4_4))
                            print(f"Inferred 4/4 time signature from measure structure (pairs of measures = 4.0 quarters)")
    except Exception as e:
        pass
    
    # If no time signature found, default to 4/4
    if not time_signatures:
        default_ts = meter.TimeSignature('4/4')
        time_signatures.append((0.0, default_ts))
        print("No time signature found in MIDI, defaulting to 4/4")
    else:
        # Sort by offset
        time_signatures.sort(key=lambda x: x[0])
        print(f"Found {len(time_signatures)} time signature(s):")
        for offset, ts in time_signatures:
            print(f"  {ts.numerator}/{ts.denominator} at offset {offset:.2f}")
    
    return time_signatures


def calculate_beat_positions(
    time_signatures: List[Tuple[float, meter.TimeSignature]],
    max_offset: float,
    subdivisions: int = 2
) -> List[Tuple[float, int, int]]:
    """
    Calculates all beat positions and subdivisions based on time signatures.
    
    Args:
        time_signatures: List of (offset, TimeSignature) tuples
        max_offset: Maximum offset to calculate beats up to
        subdivisions: Number of subdivisions per beat (e.g., 2 = half-beats, 4 = quarter-beats)
        
    Returns:
        List of tuples (position, beat_number, subdivision_number) where:
        - position: beat/subdivision position in quarter notes
        - beat_number: which beat in the measure (1-indexed)
        - subdivision_number: which subdivision (0 = main beat, 1+ = subdivisions)
    """
    beat_positions = []
    
    if not time_signatures:
        return beat_positions
    
    # Process each time signature region
    for i, (start_offset, ts) in enumerate(time_signatures):
        # Determine end offset (start of next time signature, or max_offset)
        if i + 1 < len(time_signatures):
            end_offset = time_signatures[i + 1][0]
        else:
            end_offset = max_offset
        
        # Calculate beat duration in quarter notes
        if ts.denominator == 4:
            beat_duration = 1.0  # Quarter note = 1 quarter note
        elif ts.denominator == 8:
            beat_duration = 1.5  # Dotted quarter = 1.5 quarter notes (compound time)
        elif ts.denominator == 2:
            beat_duration = 2.0  # Half note = 2 quarter notes
        else:
            # Default: assume quarter note beat
            beat_duration = 1.0
        
        # Calculate subdivision duration
        subdivision_duration = beat_duration / subdivisions
        
        # Calculate beats per measure
        beats_per_measure = ts.numerator
        
        # Generate beat positions for this time signature region
        current_pos = start_offset
        beat_counter = 0  # Track which beat we're on (0-indexed within measure)
        
        while current_pos < end_offset:
            # Calculate which beat in the measure (1-indexed)
            beat_in_measure = (beat_counter % beats_per_measure) + 1
            
            # Add this beat position
            beat_positions.append((current_pos, beat_in_measure, 0))
            
            # Add subdivisions
            for sub in range(1, subdivisions):
                sub_pos = current_pos + (sub * subdivision_duration)
                if sub_pos < end_offset:
                    beat_positions.append((sub_pos, beat_in_measure, sub))
            
            # Move to next beat
            current_pos += beat_duration
            beat_counter += 1
    
    # Sort by position
    beat_positions.sort(key=lambda x: x[0])
    
    print(f"Calculated {len(beat_positions)} beat/subdivision positions")
    return beat_positions


def visualize_beats(
    midi_file_path: str,
    subdivisions: int = 2,
    playback_speed: float = 1.0
):
    """
    Visualizes beat detection with vertical bars that illuminate on beats.
    
    Args:
        midi_file_path: Path to MIDI file
        subdivisions: Number of subdivisions per beat
        playback_speed: Speed multiplier for playback (1.0 = normal)
    """
    print(f"Loading MIDI file: {midi_file_path}")
    midi_stream = converter.parse(midi_file_path)
    
    # Detect time signatures
    print("\nDetecting time signatures...")
    print(f"MIDI stream type: {type(midi_stream)}")
    print(f"Number of parts: {len(midi_stream.parts) if hasattr(midi_stream, 'parts') else 'N/A'}")
    
    time_signatures = detect_time_signatures(midi_stream)
    
    if not time_signatures:
        print("Error: Could not determine time signature")
        return
    
    # Determine which time signature to use for visualization
    # Prefer 4/4 if detected, otherwise use the one that covers the most duration
    if len(time_signatures) == 1:
        # Only one time signature, use it
        primary_ts = time_signatures[0][1]
        primary_offset = time_signatures[0][0]
    else:
        # Multiple time signatures - prioritize 4/4, then find the one that covers the most duration
        # First, check if there's a 4/4 time signature
        four_four_ts = None
        four_four_offset = None
        for offset, ts in time_signatures:
            if ts.numerator == 4 and ts.denominator == 4:
                four_four_ts = ts
                four_four_offset = offset
                break
        
        if four_four_ts:
            # Use 4/4 if found
            primary_ts = four_four_ts
            primary_offset = four_four_offset
            print(f"\nUsing 4/4 time signature (preferred)")
        else:
            # Calculate duration covered by each time signature
            try:
                max_offset = midi_stream.duration.quarterLength
            except:
                max_offset = 1000.0  # Large default
            
            ts_durations = []
            for i, (offset, ts) in enumerate(time_signatures):
                if i + 1 < len(time_signatures):
                    duration = time_signatures[i + 1][0] - offset
                else:
                    # Last time signature - use max_offset
                    duration = max_offset - offset
                ts_durations.append((duration, offset, ts))
            
            # Sort by duration (descending) and use the one with longest duration
            ts_durations.sort(key=lambda x: x[0], reverse=True)
            primary_duration, primary_offset, primary_ts = ts_durations[0]
            
            print(f"\nTime signature durations:")
            for duration, offset, ts in ts_durations:
                print(f"  {ts.numerator}/{ts.denominator} at offset {offset:.2f}: {duration:.2f} quarters")
            print(f"Using {primary_ts.numerator}/{primary_ts.denominator} (longest duration: {primary_duration:.2f} quarters)")
    
    beats_per_measure = primary_ts.numerator
    # Number of bars should equal beats per measure (one bar per beat)
    total_bars = beats_per_measure
    
    print(f"\nPrimary time signature for visualization: {primary_ts.numerator}/{primary_ts.denominator}")
    print(f"Beats per measure: {beats_per_measure}")
    print(f"Subdivisions per beat: {subdivisions} (used for beat detection, not displayed)")
    print(f"Total bars to display: {total_bars} (one bar per beat)")
    
    # Verify the time signature makes sense
    if beats_per_measure <= 0 or beats_per_measure > 16:
        print(f"WARNING: Unusual beats per measure ({beats_per_measure}). Time signature may be incorrect.")
    
    # Calculate maximum offset (use a reasonable duration, e.g., 60 seconds worth)
    # Or calculate from actual MIDI duration
    try:
        # Try to get duration from MIDI
        duration = midi_stream.duration.quarterLength
        max_offset = duration
    except:
        max_offset = 100.0  # Default to 100 quarter notes
    
    # Calculate beat positions
    print("\nCalculating beat positions...")
    beat_positions = calculate_beat_positions(time_signatures, max_offset, subdivisions)
    
    if not beat_positions:
        print("Error: No beat positions calculated")
        return
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, total_bars)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Beat Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    ax.set_title(f'Beat Detection Visualization - {primary_ts.numerator}/{primary_ts.denominator} Time', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create bars
    bar_width = 0.8
    bars = []
    bar_colors = []
    
    for i in range(total_bars):
        # Each bar represents one beat (beat_num is 1-indexed)
        beat_num = i + 1
        
        # Color: Beat 1 is brightest, other beats are medium
        if beat_num == 1:
            # First beat of measure - brightest
            color = 'gold'
            alpha = 0.3
        else:
            # Other main beats - medium
            color = 'lightblue'
            alpha = 0.2
        
        rect = Rectangle((i, 0), bar_width, 1, 
                       facecolor=color, edgecolor='black', 
                       linewidth=1.5, alpha=alpha)
        ax.add_patch(rect)
        bars.append(rect)
        bar_colors.append((color, alpha))
    
    # Add labels for beat numbers (one label per bar)
    for i in range(total_bars):
        beat_num = i + 1
        ax.text(i + bar_width/2, -0.1, f'Beat {beat_num}', 
               ha='center', va='top', fontsize=9, fontweight='bold')
    
    # Add measure divider line
    ax.axvline(x=total_bars, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Measure End')
    ax.text(total_bars, 0.5, '|', ha='center', va='center', fontsize=20, fontweight='bold', color='red')
    
    # Animation variables
    current_time = [0.0]  # Use list to allow modification in nested function
    playback_active = [True]
    animation_obj = [None]
    
    # Get tempo for timing
    tempo = 120  # Default BPM
    try:
        tempo_marks = midi_stream.flatten().getElementsByClass('TempoIndication')
        if tempo_marks:
            tempo = tempo_marks[0].getQuarterBPM()
    except:
        pass
    
    seconds_per_quarter = 60.0 / tempo / playback_speed
    print(f"\nTempo: {tempo:.1f} BPM (playback speed: {playback_speed}x)")
    print(f"Seconds per quarter note: {seconds_per_quarter:.3f}")
    
    # Try to load audio playback libraries
    audio_available = False
    try:
        import mido
        import numpy as np
        import sounddevice as sd
        audio_available = True
        print("Audio playback libraries available - music will play during visualization")
    except ImportError as e:
        print(f"Warning: Audio libraries not available ({e})")
        print("Install with: pip install mido numpy sounddevice")
        print("Visualization will continue without audio playback")
    
    def update_animation(frame):
        """Update the visualization based on current time."""
        if not playback_active[0]:
            return bars
        
        # Find current beat position
        current_pos = current_time[0]
        
        # Find the closest beat position
        active_bar_idx = None
        min_distance = float('inf')
        
        for beat_pos, beat_num, sub_num in beat_positions:
            distance = abs(beat_pos - current_pos)
            # Only match on main beats (sub_num == 0) or allow subdivisions if close enough
            if distance < min_distance and distance < 0.1:  # Within tolerance
                min_distance = distance
                # Calculate which bar this corresponds to
                # beat_num is 1-indexed, so bar index = beat_num - 1
                # We only show main beats, so ignore subdivisions for bar selection
                if sub_num == 0:  # Only match on main beats
                    bar_idx = beat_num - 1
                    # Wrap around if we're past the first measure
                    bar_idx = bar_idx % total_bars
                    active_bar_idx = bar_idx
        
        # Update bar colors
        for i, (rect, (base_color, base_alpha)) in enumerate(zip(bars, bar_colors)):
            if i == active_bar_idx:
                # Illuminate this bar
                rect.set_facecolor('red')
                rect.set_alpha(0.9)
            else:
                # Reset to base color
                rect.set_facecolor(base_color)
                rect.set_alpha(base_alpha)
        
        # Update title with current time
        # Find the active beat info
        current_measure = 1
        current_beat = 1
        if active_bar_idx is not None:
            # Calculate from bar index (bar index is 0-indexed, beat is 1-indexed)
            current_beat = active_bar_idx + 1
            # For measure calculation, find the beat position that matches
            for beat_pos, beat_num, sub_num in beat_positions:
                if abs(beat_pos - current_pos) < 0.1 and sub_num == 0:
                    current_beat = beat_num
                    # Estimate measure from position
                    # Calculate measure duration based on time signature
                    # Find the active time signature at this position
                    active_ts = primary_ts
                    for offset, ts in time_signatures:
                        if beat_pos >= offset:
                            active_ts = ts
                        else:
                            break
                    
                    if active_ts.denominator == 4:
                        beat_duration = 1.0
                    elif active_ts.denominator == 8:
                        beat_duration = 1.5
                    elif active_ts.denominator == 2:
                        beat_duration = 2.0
                    else:
                        beat_duration = 1.0
                    # Use the active time signature's beats per measure for measure calculation
                    active_beats_per_measure = active_ts.numerator
                    measure_duration = active_beats_per_measure * beat_duration
                    current_measure = int(beat_pos / measure_duration) + 1
                    break
        
        ax.set_title(f'Beat Detection - Measure {current_measure}, Beat {current_beat} | '
                    f'Time: {current_pos:.2f} quarters ({current_pos * seconds_per_quarter:.2f}s)',
                    fontsize=14, fontweight='bold')
        
        return bars
    
    # Audio playback function
    def play_midi_audio():
        """Play MIDI audio using the same method as chord_transition_analyzer.py"""
        if not audio_available:
            return
        
        try:
            # Read MIDI file
            mid = mido.MidiFile(midi_file_path)
            
            # Get tempo from MIDI file (default 500000 microseconds per beat = 120 BPM)
            tempo_microseconds = 500000  # Default: 120 BPM
            try:
                # Find first tempo change in MIDI file
                for track in mid.tracks:
                    current_time = 0
                    for msg in track:
                        current_time += msg.time
                        if msg.type == 'set_tempo':
                            tempo_microseconds = msg.tempo
                            break
                    if tempo_microseconds != 500000:
                        break
            except:
                pass
            
            # Convert tempo to BPM for display
            tempo_bpm = 60000000.0 / tempo_microseconds
            
            # Audio synthesis parameters
            SAMPLE_RATE = 44100
            
            # ADSR envelope parameters (in seconds) - smoother transitions
            ATTACK_TIME = 0.02   # Slightly longer attack for smoother onset
            DECAY_TIME = 0.08    # Longer decay for smoother transition
            SUSTAIN_LEVEL = 0.6  # Lower sustain to reduce clipping
            RELEASE_TIME = 0.15  # Longer release for smoother fade
            
            def midi_note_to_freq(note):
                """Convert MIDI note number to frequency in Hz"""
                return 440.0 * (2.0 ** ((note - 69) / 12.0))
            
            def generate_smooth_waveform(freq, duration_samples, sample_rate):
                """Generate a smoother waveform with gentle harmonics to reduce harshness"""
                t = np.arange(duration_samples) / sample_rate
                phase = 2 * np.pi * freq * t
                
                # Use a combination of sine waves with gentle harmonics for warmer sound
                # Fundamental + slight 2nd harmonic (octave) + slight 3rd harmonic
                wave = (np.sin(phase) * 0.8 + 
                       np.sin(phase * 2) * 0.1 + 
                       np.sin(phase * 3) * 0.05)
                
                return wave
            
            print(f"Pre-rendering MIDI to audio buffer (tempo: {tempo_bpm:.1f} BPM)...", flush=True)
            
            # Collect all note events from all tracks with absolute times
            merged_track = mido.merge_tracks(mid.tracks)
            
            events = []
            current_time_seconds = 0.0
            current_tempo = tempo_microseconds
            
            for msg in merged_track:
                # Convert delta time (in ticks) to seconds
                delta_seconds = mido.tick2second(msg.time, mid.ticks_per_beat, current_tempo)
                current_time_seconds += delta_seconds
                
                # Update tempo if tempo change message
                if msg.type == 'set_tempo':
                    current_tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    events.append(('note_on', msg.note, msg.velocity, current_time_seconds))
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    events.append(('note_off', msg.note, 0, current_time_seconds))
            
            # Sort events by time
            events.sort(key=lambda x: x[3])
            
            # Calculate total duration
            if events:
                total_duration = events[-1][3] + RELEASE_TIME + 0.5  # Add release time + padding
            else:
                total_duration = 1.0
            
            total_samples = int(total_duration * SAMPLE_RATE) + 1
            
            print(f"Rendering {total_samples} samples ({total_duration:.2f} seconds)...", flush=True)
            
            # Pre-render audio buffer
            audio_buffer = np.zeros(total_samples, dtype=np.float32)
            
            # Track active notes: {note: (start_time, start_sample, velocity, frequency)}
            active_notes = {}
            
            # Process events in chronological order
            for event_type, note, velocity, event_time in events:
                sample_index = int(event_time * SAMPLE_RATE)
                
                if event_type == 'note_on':
                    freq = midi_note_to_freq(note)
                    active_notes[note] = (event_time, sample_index, velocity, freq)
                elif event_type == 'note_off' and note in active_notes:
                    start_time, start_sample, vel, freq = active_notes[note]
                    note_duration_seconds = event_time - start_time
                    note_duration_samples = sample_index - start_sample
                    
                    if note_duration_samples > 0:
                        # Render this note
                        note_samples = note_duration_samples + int(RELEASE_TIME * SAMPLE_RATE)
                        t = np.arange(note_samples) / SAMPLE_RATE
                        
                        # Generate smoother waveform
                        wave = generate_smooth_waveform(freq, note_samples, SAMPLE_RATE)
                        
                        # Apply ADSR envelope with smooth transitions
                        envelope = np.zeros_like(t)
                        
                        # Attack phase - smooth exponential curve
                        attack_mask = t < ATTACK_TIME
                        if np.any(attack_mask):
                            attack_t = t[attack_mask] / ATTACK_TIME
                            # Smooth exponential attack curve
                            envelope[attack_mask] = 1.0 - np.exp(-attack_t * 5.0)
                        
                        # Decay phase - smooth exponential decay
                        decay_start = ATTACK_TIME
                        decay_end = ATTACK_TIME + DECAY_TIME
                        decay_mask = (t >= decay_start) & (t < decay_end)
                        if np.any(decay_mask):
                            decay_t = (t[decay_mask] - decay_start) / DECAY_TIME
                            # Smooth exponential decay to sustain level
                            envelope[decay_mask] = SUSTAIN_LEVEL + (1.0 - SUSTAIN_LEVEL) * np.exp(-decay_t * 3.0)
                        
                        # Sustain phase
                        sustain_start = decay_end
                        sustain_end = max(sustain_start, note_duration_seconds - RELEASE_TIME)
                        sustain_mask = (t >= sustain_start) & (t < sustain_end)
                        if np.any(sustain_mask):
                            envelope[sustain_mask] = SUSTAIN_LEVEL
                        
                        # Release phase - smooth exponential release
                        release_mask = t >= sustain_end
                        if np.any(release_mask):
                            release_t = (t[release_mask] - sustain_end) / RELEASE_TIME
                            # Smooth exponential release
                            envelope[release_mask] = SUSTAIN_LEVEL * np.exp(-release_t * 4.0)
                        
                        # Ensure envelope doesn't go negative
                        envelope = np.clip(envelope, 0.0, 1.0)
                        
                        # Apply velocity and mix into buffer with reduced amplitude
                        amplitude = (vel / 127.0) * 0.12  # Further reduced volume to prevent clipping
                        end_sample = min(start_sample + note_samples, total_samples)
                        slice_len = end_sample - start_sample
                        audio_buffer[start_sample:end_sample] += wave[:slice_len] * envelope[:slice_len] * amplitude
                    
                    del active_notes[note]
            
            # Handle any remaining active notes
            for note, (start_time, start_sample, vel, freq) in active_notes.items():
                note_duration_seconds = total_duration - start_time - RELEASE_TIME
                note_duration_samples = int(note_duration_seconds * SAMPLE_RATE)
                if note_duration_samples > 0:
                    t = np.arange(note_duration_samples) / SAMPLE_RATE
                    
                    # Generate smoother waveform
                    wave = generate_smooth_waveform(freq, note_duration_samples, SAMPLE_RATE)
                    
                    # Apply ADSR envelope with smooth transitions
                    envelope = np.zeros_like(t)
                    
                    # Attack phase - smooth exponential curve
                    attack_mask = t < ATTACK_TIME
                    if np.any(attack_mask):
                        attack_t = t[attack_mask] / ATTACK_TIME
                        envelope[attack_mask] = 1.0 - np.exp(-attack_t * 5.0)
                    
                    # Decay phase - smooth exponential decay
                    decay_start = ATTACK_TIME
                    decay_end = ATTACK_TIME + DECAY_TIME
                    decay_mask = (t >= decay_start) & (t < decay_end)
                    if np.any(decay_mask):
                        decay_t = (t[decay_mask] - decay_start) / DECAY_TIME
                        envelope[decay_mask] = SUSTAIN_LEVEL + (1.0 - SUSTAIN_LEVEL) * np.exp(-decay_t * 3.0)
                    
                    # Sustain phase
                    sustain_start = decay_end
                    sustain_end = max(sustain_start, note_duration_seconds - RELEASE_TIME)
                    sustain_mask = (t >= sustain_start) & (t < sustain_end)
                    if np.any(sustain_mask):
                        envelope[sustain_mask] = SUSTAIN_LEVEL
                    
                    # Release phase - smooth exponential release
                    release_mask = t >= sustain_end
                    if np.any(release_mask):
                        release_t = (t[release_mask] - sustain_end) / RELEASE_TIME
                        envelope[release_mask] = SUSTAIN_LEVEL * np.exp(-release_t * 4.0)
                    
                    # Ensure envelope doesn't go negative
                    envelope = np.clip(envelope, 0.0, 1.0)
                    
                    amplitude = (vel / 127.0) * 0.12  # Reduced volume to prevent clipping
                    end_sample = min(start_sample + note_duration_samples, total_samples)
                    slice_len = end_sample - start_sample
                    audio_buffer[start_sample:end_sample] += wave[:slice_len] * envelope[:slice_len] * amplitude
            
            # Normalize and apply gentle limiting (less aggressive than before)
            max_amplitude = np.max(np.abs(audio_buffer))
            if max_amplitude > 0:
                # Normalize to 75% max to leave headroom
                audio_buffer = audio_buffer / max_amplitude * 0.75
            
            # Apply gentle soft limiting instead of aggressive clipping
            # This prevents harsh distortion while maintaining dynamics
            threshold = 0.8
            ratio = 0.3  # Gentle compression ratio
            abs_buffer = np.abs(audio_buffer)
            over_threshold = abs_buffer > threshold
            
            if np.any(over_threshold):
                # Gentle soft knee compression
                excess = abs_buffer[over_threshold] - threshold
                compressed_excess = threshold + excess * ratio
                audio_buffer[over_threshold] = np.sign(audio_buffer[over_threshold]) * compressed_excess
            
            # Final gentle tanh limiting for safety (much gentler than before)
            audio_buffer = np.tanh(audio_buffer * 0.9) * 0.9
            
            print("Audio buffer rendered successfully. Starting playback...", flush=True)
            
            # Play pre-rendered audio
            def play_audio():
                """Play pre-rendered audio buffer"""
                try:
                    sd.play(audio_buffer, samplerate=SAMPLE_RATE, blocking=False)
                    sd.wait()  # Wait for playback to finish
                    print("Audio playback completed.", flush=True)
                except Exception as e:
                    print(f"Error during audio playback: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                # Don't set playback_active[0] = False here - let the visualization continue
            
            # Start playback in a separate thread
            audio_thread = threading.Thread(target=play_audio, daemon=True)
            audio_thread.start()
            print("MIDI playback started via pre-rendered audio.", flush=True)
            
        except Exception as e:
            print(f"Error in MIDI audio playback: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    # Create animation
    def start_playback():
        """Simulate playback by updating current_time."""
        # Start audio playback if available
        if audio_available:
            play_midi_audio()
        
        start_time = time.time()
        while playback_active[0] and current_time[0] < max_offset:
            elapsed = time.time() - start_time
            current_time[0] = elapsed / seconds_per_quarter
            
            # Update every 50ms for smooth animation
            time.sleep(0.05)
        
        playback_active[0] = False
    
    # Start playback in separate thread
    import threading
    playback_thread = threading.Thread(target=start_playback, daemon=True)
    playback_thread.start()
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update_animation, interval=50, blit=False, cache_frame_data=False, repeat=True
    )
    animation_obj[0] = ani
    
    # Add stop button
    from matplotlib.widgets import Button
    plt.subplots_adjust(bottom=0.15)
    button_ax = plt.axes([0.85, 0.02, 0.12, 0.04])
    stop_button = Button(button_ax, 'Stop', color='lightcoral', hovercolor='red')
    
    def stop_playback(event):
        playback_active[0] = False
        if animation_obj[0]:
            animation_obj[0].event_source.stop()
        plt.close(fig)
    
    stop_button.on_clicked(stop_playback)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION STARTED")
    print("=" * 70)
    print("Bars will illuminate as beats occur:")
    print(f"  - Gold bars: Beat 1 of each measure (brightest)")
    print(f"  - Blue bars: Other main beats")
    print(f"  - Gray bars: Beat subdivisions")
    print(f"  - Red bars: Currently active beat/subdivision")
    print("Close the window or click 'Stop' to exit.")
    print("=" * 70 + "\n")
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python beat_detection_test.py <midi_file_path> [subdivisions] [speed]")
        print("Example: python beat_detection_test.py letitbe.mid 2 1.0")
        print("\nArguments:")
        print("  midi_file_path: Path to MIDI file")
        print("  subdivisions: Number of subdivisions per beat (default: 2)")
        print("  speed: Playback speed multiplier (default: 1.0)")
        sys.exit(1)
    
    midi_file = sys.argv[1]
    
    if not os.path.exists(midi_file):
        print(f"MIDI file not found: {midi_file}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    subdivisions = 2
    if len(sys.argv) > 2:
        try:
            subdivisions = int(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid subdivisions value '{sys.argv[2]}', using default 2")
    
    speed = 1.0
    if len(sys.argv) > 3:
        try:
            speed = float(sys.argv[3])
        except ValueError:
            print(f"Warning: Invalid speed value '{sys.argv[3]}', using default 1.0")
    
    visualize_beats(midi_file, subdivisions=subdivisions, playback_speed=speed)


if __name__ == "__main__":
    main()

