"""
Chord Transition Diagram Generator
Chordography

This script analyzes a MIDI file to:
1. Open and parse the MIDI file
2. Determine the key signature (from metadata or analysis)
3. Extract all chords in the song
4. Identify chord transitions
5. Normalize chords to roman numeral notation
6. Create a Markov chain-style graph visualization
"""

import os
import time
import threading
import csv
import re
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict

# Required libraries (install with: pip install music21 networkx matplotlib mido numpy sounddevice)
try:
    from music21 import stream, midi, key, chord, roman, analysis, converter, meter
    import networkx as nx
    # Set matplotlib backend BEFORE importing pyplot
    import matplotlib
    matplotlib.use('TkAgg')  # Force TkAgg for Windows compatibility
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    LIBRARIES_AVAILABLE = True

    # Import standalone modules
    try:
        from chord_detection import extract_chords as extract_chords_modular
        from chord_detection import create_misc_chord
    except ImportError:
        print("Warning: chord_detection.py not found. Using fallback.")
        extract_chords_modular = None

except ImportError as e:
    import sys
    print(f"Missing required library: {e}", file=sys.stderr)
    print("Install with: pip install music21 networkx matplotlib", file=sys.stderr)
    LIBRARIES_AVAILABLE = False
    # Set dummy values to prevent NameError
    stream = None
    converter = None
    key = None
    chord = None
    roman = None
    analysis = None
    animation = None
    meter = None


def load_allowed_chords(csv_path: str = "chords.csv") -> Dict[str, set]:
    """
    Loads allowed chords from chords.csv.
    Returns a dictionary with 'major' and 'minor' sets of allowed chord labels.
    """
    allowed = {'major': set(), 'minor': set()}
    
    # Check if file exists in current directory or script directory
    if not os.path.exists(csv_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "chords.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Strict chord filtering will be disabled.")
        return allowed
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header (Scale Degree, I, ii, ...)
            try:
                next(reader)
            except StopIteration:
                pass
                
            for row in reader:
                if not row: continue
                # Row format: Mode, I, ii, iii, IV, V, vi, vii°, Extras
                if len(row) < 2: continue
                
                mode = row[0].lower().strip()
                if mode in allowed:
                    # Standard degrees (columns 1-7)
                    for item in row[1:8]:
                        if item and item.strip():
                            allowed[mode].add(item.strip())
                    
                    # Extras (last column, index 8)
                    if len(row) > 8:
                        extras = row[8].split(';')
                        for item in extras:
                            if item and item.strip():
                                allowed[mode].add(item.strip())
        
        print(f"Loaded allowed chords: {len(allowed['major'])} Major, {len(allowed['minor'])} Minor")
        return allowed
    except Exception as e:
        print(f"Error loading chords.csv: {e}")
        return allowed


def filter_chord_label(label: str, allowed_set: set) -> str:
    """
    Bins a chord label into the allowed set or 'misc'.
    Strategies:
    1. Exact match
    2. Heuristic matching (strip extensions, normalize symbols)
    3. Prefix matching against allowed set
    """
    if not allowed_set:
        return label
        
    # Normalize label (trim whitespace)
    label = label.strip()
    
    # 0. Pre-processing normalization (handle music21 variations)
    # Replace 'dim' or 'o' at end with degree symbol if mostly roman
    if label.endswith('dim') or label.endswith('o'):
        label = label.replace('dim', '°').replace('o', '°')
    
    # 1. Exact match
    if label in allowed_set:
        return label
    
    # 2. Try identifying the core Roman Numeral (Root + Accidental + Quality)
    # Regex to capture: (Accidental)(Roman)(Quality)
    # e.g. #vii°7 -> #, vii, °
    # e.g. V7b9 -> V
    # e.g. bVI -> b, VI
    
    # Pattern: Optional #/b, then Roman chars (I, V, i, v), then optional quality (°, +, o, ø)
    # We want to match the START of the string.
    roman_pattern = r'^([#b]?[IViv]+[°+øo]?)'
    match = re.match(roman_pattern, label)
    
    if match:
        core_root = match.group(1)
        
        # Check if the core root itself is allowed (e.g. "V", "iv")
        if core_root in allowed_set:
            # If we have a generic match, check if we can find a more specific one in allowed_set
            # that is also a prefix of our label (e.g. label="V7b9", allowed has "V7", match "V7")
            
            # Find all allowed chords that start with this root
            candidates = [c for c in allowed_set if c.startswith(core_root)]
            # Sort by length descending to match most specific first (V7 before V)
            candidates.sort(key=len, reverse=True)
            
            for cand in candidates:
                # Check if label starts with candidate
                # And the rest of label is just extensions/numbers/accidentals
                if label.startswith(cand):
                    return cand
            
            # If no specific extension matched, return the core root
            return core_root
            
        # Handle cases where core root extraction missed the 7 or similar standard suffixes
        # e.g. label="V7", core="V". If "V7" is allowed, use it.
        # This is covered by the prefix search above IF core_root is a prefix of allowed chords.
        # But if core_root is NOT in allowed_set (e.g. maybe "vii" is not allowed, but "vii°" is?)
        
        # Try normalizing 'o' to '°' just in case
        core_norm = core_root.replace('o', '°')
        if core_norm in allowed_set:
            return core_norm
            
    # 3. Special handling for slash chords
    if '/' in label:
        parts = label.split('/')
        if len(parts) == 2:
            # Recursively filter the left side
            # We construct a temporary allowed set of just roots from the real allowed set
            # But simpler: just try to simplify the left side to a bare roman numeral
            root_part = parts[0]
            root_match = re.match(roman_pattern, root_part)
            if root_match:
                core_root = root_match.group(1)
                # Reconstruct V/V
                slash_chord = f"{core_root}/{parts[1]}"
                if slash_chord in allowed_set:
                    return slash_chord
                # Try with normalized secondary (remove digits from secondary)
                sec_clean = re.sub(r'\d+', '', parts[1])
                slash_chord = f"{core_root}/{sec_clean}"
                if slash_chord in allowed_set:
                    return slash_chord

    # 4. Fallback: extensive cleaning
    # Remove all digits
    no_digits = re.sub(r'\d+', '', label)
    if no_digits in allowed_set:
        return no_digits
        
    # Remove digits and common extension accidentals (#, b) that appear at the END
    # This is risky as it might remove root accidentals (#IV) if regex is sloppy
    # but let's try removing anything that isn't the main roman numeral
    if match:
        core_root = match.group(1)
        if core_root in allowed_set:
            return core_root
            
    # If we are here, we failed to match
    # print(f"DEBUG: Could not bin chord '{label}' -> misc")
    return 'misc'


def open_midi_file(file_path: str) -> stream.Stream:
    """
    Opens and parses a MIDI file.
    
    Args:
        file_path: Path to the MIDI file
        
    Returns:
        music21 Stream object containing the parsed MIDI data
    """
    # Use converter.parse() which is the standard way to parse MIDI files in music21
    midi_stream = converter.parse(file_path)
    return midi_stream


def determine_key_signature(midi_stream: stream.Stream) -> key.Key:
    """
    Determines the key signature of the piece.
    First checks if key signature is provided in metadata,
    otherwise analyzes the piece to determine the key.
    
    Args:
        midi_stream: music21 Stream object
        
    Returns:
        music21 Key object representing the detected key
    """
    # First, check if Key objects are directly present in the stream
    key_objects = midi_stream.flatten().getElementsByClass(key.Key)
    if key_objects:
        # If Key object is found, use it directly
        return key_objects[0]
    
    # Second, check if key signature is explicitly provided in the MIDI file
    # Look for KeySignature objects in the stream
    key_signatures = midi_stream.flatten().getElementsByClass(key.KeySignature)
    
    if key_signatures:
        # If key signature is found, convert it to a Key object
        # The first key signature in the piece is typically the main key
        key_sig = key_signatures[0]
        # Convert KeySignature to Key
        # KeySignature.asKey() returns a Key object with the appropriate mode
        detected_key = key_sig.asKey()
        return detected_key
    
    # If no key signature is found in metadata, use music21's key analysis
    # music21 uses the Krumhansl-Schmuckler key-finding algorithm
    # which analyzes the distribution of pitches to determine the most likely key
    try:
        analyzed_key = midi_stream.analyze('key')
        return analyzed_key
    except Exception as e:
        # Fallback: use discrete analysis if analyze() fails
        print(f"Warning: Standard key analysis failed ({e}), trying alternative method...")
        try:
            analyzed_key = analysis.discrete.analyzeStream(midi_stream, 'key')
            return analyzed_key
        except Exception as e2:
            # Final fallback: default to C major if all analysis fails
            print(f"Warning: Key analysis failed ({e2}), defaulting to C major")
            return key.Key('C')


def detect_time_signatures(midi_stream: stream.Stream) -> List[Tuple[float, meter.TimeSignature]]:
    """
    Detects all time signatures and their positions in the MIDI stream.
    Checks multiple locations: flattened stream, parts, and measures.
    Also infers time signature from measure structure when metadata is incorrect.
    
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
            analyzed_ts = midi_stream.analyze('timeSignature')
            if analyzed_ts:
                time_signatures.append((0.0, analyzed_ts))
                print(f"Inferred time signature from music analysis: {analyzed_ts.numerator}/{analyzed_ts.denominator}")
        except:
            pass
    
    # Method 5: Infer time signature from measure structure
    # Check if consecutive measures suggest a different grouping (e.g., 4/4 mis-grouped as 2/4)
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
) -> List[float]:
    """
    Calculates all beat positions and subdivisions based on time signatures.
    
    Args:
        time_signatures: List of (offset, TimeSignature) tuples
        max_offset: Maximum offset to calculate beats up to
        subdivisions: Number of subdivisions per beat (e.g., 2 = half-beats, 4 = quarter-beats)
        
    Returns:
        List of beat/subdivision positions (in quarter notes)
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
        # For 4/4: beat = 1 quarter note
        # For 3/4: beat = 1 quarter note
        # For 6/8: beat = 1.5 quarter notes (dotted quarter)
        # The denominator tells us the note value: 4 = quarter note, 8 = eighth note
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
        
        # Generate beat positions for this time signature region
        current_pos = start_offset
        while current_pos < end_offset:
            # Add this beat position
            beat_positions.append(current_pos)
            
            # Add subdivisions
            for sub in range(1, subdivisions):
                sub_pos = current_pos + (sub * subdivision_duration)
                if sub_pos < end_offset:
                    beat_positions.append(sub_pos)
            
            # Move to next beat
            current_pos += beat_duration
        
        # Ensure we include the end offset if it's a beat boundary
        if end_offset <= max_offset and end_offset not in beat_positions:
            beat_positions.append(end_offset)
    
    # Sort and remove duplicates
    beat_positions = sorted(set(beat_positions))
    
    print(f"Calculated {len(beat_positions)} beat/subdivision positions")
    return beat_positions


def align_chords_to_beats(
    chords: List[chord.Chord],
    offsets: List[float],
    durations: List[float],
    beat_positions: List[float],
    tolerance: float = 0.1
) -> Tuple[List[chord.Chord], List[float], List[float]]:
    """
    Filters chords to only include those that align with beats or beat subdivisions.
    
    Args:
        chords: List of chord.Chord objects
        offsets: List of chord start offsets
        durations: List of chord durations
        beat_positions: List of beat/subdivision positions
        tolerance: Maximum distance from beat position to consider aligned (in quarter notes)
        
    Returns:
        Tuple of (filtered_chords, filtered_offsets, filtered_durations)
    """
    if not beat_positions:
        print("Warning: No beat positions calculated, returning all chords")
        return chords, offsets, durations
    
    aligned_chords = []
    aligned_offsets = []
    aligned_durations = []
    
    # Create a set for fast lookup (with tolerance)
    beat_set = set(beat_positions)
    
    for i, (ch, offset, duration) in enumerate(zip(chords, offsets, durations)):
        # Check if this chord's offset is close to any beat position
        is_aligned = False
        
        # Check exact match first
        if offset in beat_set:
            is_aligned = True
        else:
            # Check if within tolerance of any beat position
            for beat_pos in beat_positions:
                if abs(offset - beat_pos) <= tolerance:
                    is_aligned = True
                    break
        
        if is_aligned:
            aligned_chords.append(ch)
            aligned_offsets.append(offset)
            aligned_durations.append(duration)
    
    print(f"Aligned chords: {len(aligned_chords)}/{len(chords)} chords align with beats/subdivisions")
    
    return aligned_chords, aligned_offsets, aligned_durations


def extract_chords(midi_stream: stream.Stream) -> Tuple[List[chord.Chord], List[float], List[float]]:
    """
    Extracts all chords from the MIDI stream using the standalone chord_detection module.
    Filters chords to only include those at beats and beat subdivisions.
    
    Args:
        midi_stream: music21 Stream object
        
    Returns:
        Tuple of (chords, offsets, durations)
    """
    if extract_chords_modular:
        # Use the robust template matching algorithm from chord_detection.py
        # We need to determine key signature first for the template matching
        # But wait, the modular function determines key internally if passed key_sig=None
        # However, we want to be consistent with the main pipeline's key detection if possible.
        # The main pipeline calls extract_chords AFTER determine_key_signature, but
        # extract_chords signature doesn't accept key.
        # Let's let the modular function detect its own key or just use what it finds.
        
        # Actually, let's pass the key if we can, but since we can't change the signature easily
        # without breaking other things, let's just let it detect.
        # OR: We can cheat and look if key is in the stream.
        
        # Better: let's stick to the signature but internally use the modular one.
        print("Using robust template-matching chord detection from chord_detection.py...")
        chords, offsets, durations = extract_chords_modular(midi_stream, key_sig=None, csv_path="chords.csv", use_csv_filter=True)
        
        # Now filter to beats/subdivisions
        print("\nDetecting time signatures and aligning chords to beats...")
        time_signatures = detect_time_signatures(midi_stream)
        
        # Select primary time signature (prefer 4/4, otherwise use longest duration)
        if len(time_signatures) > 1:
            # Check if there's a 4/4 time signature
            four_four_ts = None
            for offset, ts in time_signatures:
                if ts.numerator == 4 and ts.denominator == 4:
                    four_four_ts = ts
                    break
            
            if four_four_ts:
                # Use 4/4 if found
                primary_ts = four_four_ts
                print(f"Using 4/4 time signature for beat alignment (preferred)")
            else:
                # Use the one that covers the most duration
                try:
                    max_offset_calc = max(offsets) + max(durations) if offsets else 100.0
                    ts_durations = []
                    for i, (offset, ts) in enumerate(time_signatures):
                        if i + 1 < len(time_signatures):
                            duration = time_signatures[i + 1][0] - offset
                        else:
                            duration = max_offset_calc - offset
                        ts_durations.append((duration, ts))
                    ts_durations.sort(key=lambda x: x[0], reverse=True)
                    primary_ts = ts_durations[0][1]
                except:
                    primary_ts = time_signatures[0][1]
        else:
            primary_ts = time_signatures[0][1]
        
        # Calculate maximum offset
        max_offset = max(offsets) + max(durations) if offsets else 100.0
        
        # Calculate beat positions (with 2 subdivisions per beat = half-beats)
        beat_positions = calculate_beat_positions(time_signatures, max_offset, subdivisions=2)
        
        # Filter chords to only those aligned with beats
        chords, offsets, durations = align_chords_to_beats(chords, offsets, durations, beat_positions, tolerance=0.1)
        
        return chords, offsets, durations
    
    else:
        # Fallback to old simple logic if module is missing
        print("Falling back to simple legacy chord detection...")
        
        # Use chordify() to reduce multi-part/track MIDI to a single staff
        chordified = midi_stream.chordify()
        
        # Extract all Chord objects from the chordified stream with their timing
        raw_chords = []
        raw_offsets = []
        raw_durations = []
        
        for ch in chordified.flatten().getElementsByClass(chord.Chord):
            raw_chords.append(ch)
            raw_offsets.append(ch.offset)
            raw_durations.append(ch.duration.quarterLength)
        
        # Process chords: preserve original chord information for better roman numeral analysis
        # Only simplify for visualization purposes (reduce to unique chord types)
        processed_chords = []
        processed_offsets = []
        processed_durations = []
        
        # Track unique chord signatures to reduce duplicates
        chord_signatures = {}  # Maps chord signature to representative chord
        
        for i, ch in enumerate(raw_chords):
            # Get unique pitch classes (remove octave information but keep all notes)
            pitch_classes = tuple(sorted(set([p.pitchClass for p in ch.pitches])))
            
            # Use original chord if we haven't seen this pitch class combination
            # This preserves all notes for better roman numeral analysis
            if pitch_classes not in chord_signatures:
                # Use the original chord, but ensure it's in a consistent form
                # Create a new chord with sorted pitch classes for consistency
                unique_pitches = sorted(set(ch.pitches), key=lambda p: p.pitchClass)
                representative_chord = chord.Chord(unique_pitches)
                chord_signatures[pitch_classes] = representative_chord
                processed_chords.append(representative_chord)
            else:
                # Reuse the representative chord
                processed_chords.append(chord_signatures[pitch_classes])
            
            # Store timing information
            processed_offsets.append(raw_offsets[i])
            processed_durations.append(raw_durations[i])
        
        # Filter to beats/subdivisions
        print("\nDetecting time signatures and aligning chords to beats...")
        time_signatures = detect_time_signatures(midi_stream)
        
        # Select primary time signature (prefer 4/4, otherwise use longest duration)
        if len(time_signatures) > 1:
            # Check if there's a 4/4 time signature
            four_four_ts = None
            for offset, ts in time_signatures:
                if ts.numerator == 4 and ts.denominator == 4:
                    four_four_ts = ts
                    break
            
            if four_four_ts:
                # Use 4/4 if found
                primary_ts = four_four_ts
                print(f"Using 4/4 time signature for beat alignment (preferred)")
            else:
                # Use the one that covers the most duration
                try:
                    max_offset_calc = max(processed_offsets) + max(processed_durations) if processed_offsets else 100.0
                    ts_durations = []
                    for i, (offset, ts) in enumerate(time_signatures):
                        if i + 1 < len(time_signatures):
                            duration = time_signatures[i + 1][0] - offset
                        else:
                            duration = max_offset_calc - offset
                        ts_durations.append((duration, ts))
                    ts_durations.sort(key=lambda x: x[0], reverse=True)
                    primary_ts = ts_durations[0][1]
                except:
                    primary_ts = time_signatures[0][1]
        else:
            primary_ts = time_signatures[0][1]
        
        # Calculate maximum offset
        max_offset = max(processed_offsets) + max(processed_durations) if processed_offsets else 100.0
        
        # Calculate beat positions (with 2 subdivisions per beat = half-beats)
        beat_positions = calculate_beat_positions(time_signatures, max_offset, subdivisions=2)
        
        # Filter chords to only those aligned with beats
        processed_chords, processed_offsets, processed_durations = align_chords_to_beats(
            processed_chords, processed_offsets, processed_durations, beat_positions, tolerance=0.1
        )
        
        # If we have more than 25 unique chords, reduce by grouping similar chords
        unique_chords = list(set(processed_chords))
        
        if len(unique_chords) > 25:
            # Count frequency of each unique chord
            chord_counts = Counter(processed_chords)
            
            # Keep the 25 most common chords
            top_25_chords = [ch for ch, _ in chord_counts.most_common(25)]
            
            # Map all chords to the closest match in top 25
            # Match based on root and similar pitch class content
            final_chords = []
            final_offsets = []
            final_durations = []
            
            for i, ch in enumerate(processed_chords):
                if ch in top_25_chords:
                    final_chords.append(ch)
                else:
                    # Find the closest match (same root, most similar pitch classes)
                    root_pc = ch.root().pitchClass
                    ch_pcs = set([p.pitchClass for p in ch.pitches])
                    
                    # Find chords with same root
                    matching_chords = [c for c in top_25_chords if c.root().pitchClass == root_pc]
                    
                    if matching_chords:
                        # Use the one with most similar pitch classes
                        best_match = max(matching_chords, 
                                       key=lambda c: len(set([p.pitchClass for p in c.pitches]) & ch_pcs))
                        final_chords.append(best_match)
                    else:
                        # Use the most common chord overall
                        final_chords.append(top_25_chords[0])
                
                # Preserve timing information
                final_offsets.append(processed_offsets[i])
                final_durations.append(processed_durations[i])
            
            return final_chords, final_offsets, final_durations
        
        return processed_chords, processed_offsets, processed_durations


def determine_chord_transitions(chords: List[chord.Chord]) -> List[Tuple[chord.Chord, chord.Chord]]:
    """
    Determines all chord transitions in the song.
    A transition occurs when one chord is followed by another.
    
    Args:
        chords: List of chord.Chord objects
        
    Returns:
        List of tuples representing (from_chord, to_chord) transitions
    """
    transitions = []
    
    # Iterate through consecutive pairs of chords
    # For each chord (except the last), create a transition to the next chord
    for i in range(len(chords) - 1):
        from_chord = chords[i]
        to_chord = chords[i + 1]
        transitions.append((from_chord, to_chord))
    
    return transitions


def correct_chord_spelling(ch: chord.Chord, key_sig: key.Key) -> chord.Chord:
    """
    Corrects the enharmonic spelling of a chord's pitches to match the key signature.
    This fixes issues where MIDI uses default spellings (e.g. G#) that don't match 
    the key (e.g. Ab Major), causing Roman Numeral analysis to fail (identifying I as #vii).
    """
    new_pitches = []
    changed = False
    
    for p in ch.pitches:
        # Check if current spelling matches a scale degree
        if key_sig.getScaleDegreeFromPitch(p) is not None:
            new_pitches.append(p)
            continue
            
        # Try the primary enharmonic equivalent
        enh = p.getEnharmonic()
        if key_sig.getScaleDegreeFromPitch(enh) is not None:
            new_pitches.append(enh)
            changed = True
            continue
            
        # Keep original if neither fits (chromatic note)
        # Note: could be improved by checking which spelling makes more sense harmonically
        new_pitches.append(p)
    
    if changed:
        return chord.Chord(new_pitches)
    return ch


def normalize_to_roman_numerals(
    chords: List[chord.Chord], 
    key_signature: key.Key
) -> List[str]:
    """
    Normalizes all chords to roman numeral notation based on the key signature.
    
    Args:
        chords: List of chord.Chord objects
        key_signature: music21 Key object
        
    Returns:
        List of roman numeral strings (e.g., 'I', 'ii', 'V7', etc.)
    """
    roman_numerals = []
    failed_count = 0
    for ch in chords:
        try:
            # Correct enharmonic spelling to match key
            # This is crucial for MIDI files where notes might be spelled "wrong" for the inferred key
            ch_corrected = correct_chord_spelling(ch, key_signature)
            
            # Ensure chord has at least 2 notes for proper analysis
            if len(ch_corrected.pitches) < 2:
                # Single note - try to create a triad based on the note
                root_pitch = ch_corrected.pitches[0]
                # Create a major triad (most common)
                triad = chord.Chord([root_pitch, root_pitch.transpose(4), root_pitch.transpose(7)])
                rn = roman.romanNumeralFromChord(triad, key_signature)
            else:
                rn = roman.romanNumeralFromChord(ch_corrected, key_signature)
            roman_numerals.append(str(rn.figure))
        except Exception as e:
            # If roman numeral conversion fails, try to analyze the chord differently
            failed_count += 1
            try:
                # Try with root position chord using corrected chord
                ch_root = ch_corrected.root()
                # Create a simplified version with root, third, and fifth
                root_pc = ch_root.pitchClass
                pitch_classes = sorted(set([p.pitchClass for p in ch.pitches]))
                
                # Try to build a triad
                third_pc = None
                fifth_pc = None
                for pc in pitch_classes:
                    interval = (pc - root_pc) % 12
                    if interval == 3 or interval == 4:  # Minor or major third
                        third_pc = pc
                    elif interval == 7:  # Perfect fifth
                        fifth_pc = pc
                
                if third_pc is not None and fifth_pc is not None:
                    # Build triad
                    triad_pitches = [ch_root]
                    if third_pc != root_pc:
                        triad_pitches.append(ch_root.transpose((third_pc - root_pc) % 12))
                    if fifth_pc != root_pc:
                        triad_pitches.append(ch_root.transpose((fifth_pc - root_pc) % 12))
                    triad = chord.Chord(triad_pitches)
                    rn = roman.romanNumeralFromChord(triad, key_signature)
                    roman_numerals.append(str(rn.figure))
                else:
                    # Fallback to chord name
                    chord_name = ch_root.name
                    roman_numerals.append(chord_name)
            except:
                # Final fallback: use root note name
                try:
                    root_name = ch.root().name
                    roman_numerals.append(root_name)
                except:
                    roman_numerals.append('?')
    
    if failed_count > 0:
        print(f"Warning: {failed_count} chords could not be converted to roman numerals", flush=True)
    
    return roman_numerals


def create_transition_graph(
    chords: List[chord.Chord],
    transitions: List[Tuple[chord.Chord, chord.Chord]],
    key_signature: key.Key
) -> Tuple[nx.DiGraph, Dict[chord.Chord, str], List[str]]:
    """
    Creates a directed graph representing chord transitions.
    Nodes represent chords (in roman numeral notation),
    edges represent transitions with weights (counts).
    
    Args:
        chords: List of chord.Chord objects (in order)
        transitions: List of (from_chord, to_chord) tuples
        key_signature: music21 Key object (for converting transitions)
        
    Returns:
        Tuple of (networkx DiGraph, chord_to_label mapping, list of chord labels in order)
    """
    G = nx.DiGraph()
    
    # Load allowed chords
    allowed_chords = load_allowed_chords()
    
    # Select mode
    mode = 'major'
    if key_signature and key_signature.mode == 'minor':
        mode = 'minor'
    allowed_set = allowed_chords.get(mode, set())
    
    # Convert chords to labels (roman numerals or chord names)
    chord_to_label = {}
    chord_labels = []
    
    for ch in chords:
        # Create a unique identifier for the chord
        # Include bass in signature if available, or just pitches
        chord_key = tuple(sorted(set([p.pitchClass for p in ch.pitches])))
        
        if chord_key not in chord_to_label:
            try:
                # Use pre-calculated label from modular detector if available
                if hasattr(ch, 'roman_label'):
                    label = ch.roman_label
                else:
                    # Correct enharmonic spelling to match key
                    ch_corrected = correct_chord_spelling(ch, key_signature)
                    rn = roman.romanNumeralFromChord(ch_corrected, key_signature)
                    label = str(rn.figure)
                
                # Filter/Bin the label
                # Only filter if we have a valid allowed set
                # STRICT FILTERING DISABLED AS PER REQUEST
                # if allowed_set:
                #    label = filter_chord_label(label, allowed_set)
                
            except Exception:
                # Fallback to chord name
                label = ', '.join(sorted(set(ch.pitchNames)))
            
            chord_to_label[chord_key] = label
            chord_labels.append(label)
        else:
            chord_labels.append(chord_to_label[chord_key])
    
    # Count transitions between chord labels
    transition_counts = Counter()
    for from_chord, to_chord in transitions:
        from_key = tuple(sorted(set([p.pitchClass for p in from_chord.pitches])))
        to_key = tuple(sorted(set([p.pitchClass for p in to_chord.pitches])))
        
        from_label = chord_to_label[from_key]
        to_label = chord_to_label[to_key]
        
        transition_counts[(from_label, to_label)] += 1
    
    # Add nodes and edges with weights
    for (from_label, to_label), count in transition_counts.items():
        # Exclude self-loops (transitions to the same chord)
        if from_label == to_label:
            continue
            
        if G.has_edge(from_label, to_label):
            G[from_label][to_label]['weight'] += count
        else:
            G.add_edge(from_label, to_label, weight=count)
    
    return G, chord_to_label, chord_labels


def visualize_graph(
    graph: nx.DiGraph,
    chords: List[chord.Chord],
    chord_labels: List[str],
    chord_offsets: List[float],
    chord_durations: List[float],
    midi_file_path: str,
    key_signature: key.Key,
    output_path: str = "chord_transition_graph.png"
):
    """
    Visualizes the chord transition graph as a Markov chain-style diagram with real-time playback.
    Shows nodes (chords) and edges (transitions) with transition counts as labels.
    Plays MIDI via Python and highlights the currently playing chord node.
    
    Args:
        graph: networkx DiGraph object
        chords: List of chord.Chord objects in chronological order
        chord_labels: List of chord label strings in chronological order
        chord_offsets: List of float offsets (in quarter notes) for when each chord starts
        chord_durations: List of float durations (in quarter notes) for how long each chord lasts
        midi_file_path: Path to the MIDI file for playback
        output_path: Path to save the static visualization
    """
    # Ensure we're using an interactive backend
    import matplotlib
    print(f"Current matplotlib backend: {matplotlib.get_backend()}", flush=True)
    
    # Force TkAgg backend for Windows compatibility
    try:
        matplotlib.use('TkAgg')
        print(f"Set backend to: {matplotlib.get_backend()}", flush=True)
    except Exception as e:
        print(f"Warning: Could not set TkAgg backend: {e}", flush=True)
        # Try Qt5Agg as fallback
        try:
            matplotlib.use('Qt5Agg')
            print(f"Set backend to: {matplotlib.get_backend()}", flush=True)
        except:
            pass
    
    # Create figure and axis
    print("Creating matplotlib figure...", flush=True)
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        print("Figure created successfully.", flush=True)
    except Exception as e:
        print(f"ERROR creating figure: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    # Calculate layout (positions of nodes) - Custom "Sacred Ring" Layout
    print("Calculating 'Sacred Ring' layout...", flush=True)
    import math
    
    pos = {}
    
    # 1. Define the Sacred Angles (Clockwise starting from 12 o'clock, or arranged circle of fifths?)
    # Let's use scale degree order (I, ii, iii, IV, V, vi, vii) clockwise
    # 7 points on a circle = 360/7 degrees per slice
    
    # Map scale degrees to angles (radians)
    # 0 is East, pi/2 is North.
    # Let's put I at Top (pi/2)
    
    degree_order = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii']
    # Normalize input to match these (case insensitive?)
    # Actually, let's just look at the root.
    
    # Angles: I at 90 deg (pi/2), others spaced evenly clockwise
    # 90 deg = pi/2
    # clockwise means subtracting 2pi/7
    
    base_radius = 5.5  # Increased base radius for more spacing
    
    def get_roman_root(label):
        """Extract the root roman numeral (e.g., 'V7' -> 'V', 'ii6' -> 'ii', '##ivb6b42' -> 'iv')"""
        if label == "misc" or label == "[MISC]":
            return "misc"
        
        import re
        
        # Handle secondary dominants first (V/V -> use the first part)
        if '/' in label:
            label = label.split('/')[0]
        
        # Remove all accidentals (#, b) and numbers from the start
        # Pattern: match roman numerals (i, v, I, V, and combinations like ii, iii, IV, etc.)
        # Roman numerals can be: I, II, III, IV, V, VI, VII, i, ii, iii, iv, v, vi, vii
        # Match at the start: optional # or b, then roman numeral pattern
        
        # First, strip leading accidentals
        label_clean = label.lstrip('#b')
        
        # Match roman numeral pattern at the start
        # Pattern: Put IV and VI before I{1,3} to ensure they match correctly
        # Order matters in regex alternation - more specific patterns should come first
        roman_pattern = r'^(IV|VI{0,2}|I{1,3}|iv|vi{0,2}|V|v)'
        match = re.match(roman_pattern, label_clean, re.IGNORECASE)
        
        if match:
            root = match.group(1)
            # Normalize case based on whether it starts with uppercase
            if root[0].isupper():
                # Major: ensure proper case (I, II, III, IV, V, VI, VII)
                root = root.upper()
                if root == 'II':
                    root = 'ii'  # ii is lowercase in major
                elif root == 'III':
                    root = 'iii'  # iii is lowercase in major
                elif root == 'VI':
                    root = 'vi'  # vi is lowercase in major
                elif root == 'VII':
                    root = 'vii'  # vii is lowercase in major
            else:
                # Minor: keep as is or normalize
                root = root.lower()
            return root
        
        # Fallback: try to extract by removing common suffixes
        # Be careful: don't remove '4' from 'IV' - only remove inversion numbers like '64', '43', etc.
        # Remove chord extensions and inversions, but preserve roman numeral structure
        clean = label.lstrip('#b')
        # Remove common suffixes that come AFTER the roman numeral
        # Use regex to remove patterns like 7, 6, 64, 43, 42, 9, etc. at the end
        clean = re.sub(r'[°+ø]', '', clean)  # Remove quality symbols
        clean = re.sub(r'(64|43|42|65|63|54|53|52)$', '', clean)  # Remove inversion numbers
        clean = re.sub(r'[679]$', '', clean)  # Remove single extension numbers at end only
        
        # Try to match again (use same pattern with IV first)
        roman_pattern = r'^(IV|VI{0,2}|I{1,3}|iv|vi{0,2}|V|v)'
        match = re.match(roman_pattern, clean, re.IGNORECASE)
        if match:
            root = match.group(1)
            if root[0].isupper():
                root = root.upper()
                if root in ['II', 'III', 'VI', 'VII']:
                    root = root.lower()
            else:
                root = root.lower()
            return root
        
        # Final fallback: return the cleaned label (might be partial but better than nothing)
        return clean if clean else label.lstrip('#b')
    
    # Canonical mapping for position lookup - KEY-AWARE
    # Determine if major or minor
    is_major = key_signature.mode == 'major'
    
    if is_major:
        # Major scale: I, ii, iii, IV, V, vi, vii°
        canonical_map = {
            'I': 0, 'ii': 1, 'iii': 2, 'IV': 3, 'V': 4, 'vi': 5, 'vii': 6,
            # Also handle uppercase variants
            'II': 1, 'III': 2, 'VI': 5, 'VII': 6
        }
        # Vanilla chords for major (at base_radius)
        vanilla_chords = {'I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii', 'vii°', 'viidim'}
    else:
        # Minor scale: i, ii°, III, iv, v, VI, VII
        canonical_map = {
            'i': 0, 'ii': 1, 'III': 2, 'iv': 3, 'v': 4, 'VI': 5, 'VII': 6,
            # Also handle variants
            'II': 1, 'IV': 3, 'V': 4
        }
        # Vanilla chords for minor (at base_radius)
        vanilla_chords = {'i', 'ii', 'ii°', 'iidim', 'III', 'iv', 'v', 'VI', 'VII'}
    
    node_groups = defaultdict(list)
    
    for node in graph.nodes():
        node_str = str(node)
        root_str = get_roman_root(node_str)
        
        # Check for misc first
        if 'misc' in node_str.lower() or root_str == 'misc':
            node_groups['misc'].append(node)
            continue
        
        # Clean up flat/sharp prefixes for positioning (bVII -> VII, but preserve case)
        core_degree = root_str.replace('b', '').replace('#', '')
        
        # Try exact match first
        if core_degree in canonical_map:
            idx = canonical_map[core_degree]
            node_groups[idx].append(node)
        else:
            # Try case-insensitive match
            found = False
            for k, idx in canonical_map.items():
                if core_degree.lower() == k.lower():
                    node_groups[idx].append(node)
                    found = True
                    break
            
            if not found:
                # Try prefix matching (e.g., 'vii' matches 'vii°')
                for k, idx in canonical_map.items():
                    if core_degree.lower().startswith(k.lower()) or k.lower().startswith(core_degree.lower()):
                        node_groups[idx].append(node)
                        found = True
                        break
            
            if not found:
                # Last resort: map to misc
                print(f"DEBUG: Could not map node '{node_str}' (root: '{root_str}', core: '{core_degree}') to scale degree, mapping to misc", flush=True)
                node_groups['misc'].append(node)
    
    # Debug: Print summary of node groups
    print(f"\nDEBUG: Node grouping summary:", flush=True)
    for idx in range(7):
        if idx in node_groups:
            node_names = [str(n) for n in node_groups[idx]]
            print(f"  Scale degree {idx}: {len(node_groups[idx])} nodes - {node_names[:5]}...", flush=True)
    if 'misc' in node_groups:
        print(f"  Misc: {len(node_groups['misc'])} nodes", flush=True)
    print("", flush=True)

    # Calculate positions
    start_angle = math.pi / 2 # Top
    angle_step = 2 * math.pi / 7 # 7 slices
    
    for key_idx, nodes in node_groups.items():
        if key_idx == 'misc':
            # Place misc nodes at the center of the circle
            for i, n in enumerate(nodes):
                if len(nodes) == 1:
                    # Single misc node at exact center
                    pos[n] = (0, 0)
                else:
                    # Multiple misc nodes in a small circle at center
                    r = 0.3  # Very small radius for center cluster
                    theta = (2 * math.pi / len(nodes)) * i if nodes else 0
                    pos[n] = (r * math.cos(theta), r * math.sin(theta))
            continue
            
        # For standard degrees
        # Calculate base angle
        # Note: clockwise = subtract angle
        angle = start_angle - (key_idx * angle_step)
        
        # Sort nodes: Vanilla (len usually shortest) -> closest to base_radius
        # Complex -> further out or in?
        # Requirement: "modifications should be radially inward/outward"
        
        # Let's sort by string length as a proxy for complexity (I vs I64 vs V7)
        # Ideally: Base triad at base_radius. Inversions inward? Extensions outward?
        
        sorted_nodes = sorted(nodes, key=len)
        
        # Even radial spacing: calculate spacing interval
        num_nodes = len(sorted_nodes)
        
        # Calculate spacing: if only 1 node, put at base. If many, space them out but ensure no overlap
        # Node size is about 900 points. In data coordinates, that's roughly 0.5 units diameter
        node_diameter = 0.5
        
        # Find vanilla chord index (should be at base_radius)
        vanilla_idx = None
        for i, n in enumerate(sorted_nodes):
            node_str = str(n)
            # Extract root without any extensions/modifications
            root_only = get_roman_root(node_str)
            
            # Check multiple ways to identify vanilla chord
            # Vanilla = base triad with no extensions (no 7, 6, 9, inversions, etc.)
            is_vanilla = (
                # Exact match in vanilla_chords
                node_str in vanilla_chords or 
                # Root matches vanilla (and node has no extensions)
                (root_only in vanilla_chords and 
                 not any(ext in node_str for ext in ['7', '9', '6', '4', '2', '11', '13']) and
                 '64' not in node_str and '43' not in node_str and '42' not in node_str) or
                # Simple triad (just letters, no numbers, matches canonical)
                (len(node_str) <= 3 and 
                 node_str.replace('°', '').replace('dim', '').replace('o', '').isalpha() and 
                 root_only in canonical_map and
                 root_only in vanilla_chords)
            )
            if is_vanilla:
                vanilla_idx = i
                break
        
        # If no vanilla found, use the shortest/simplest node (likely the base chord)
        if vanilla_idx is None:
            vanilla_idx = 0
        
        # Position nodes with even spacing
        # Vanilla chord at base_radius, others evenly spaced outward
        for i, n in enumerate(sorted_nodes):
            # Calculate distance from vanilla
            dist_idx = i - vanilla_idx
            
            # Use dynamic spacing to prevent overlap
            # Increased spacing to prevent node overlap
            # Minimum spacing should be at least node_diameter + generous padding
            spacing = max(2.0, node_diameter * 4.0)  # Ensure at least 2.0 units between node centers for bigger nodes
            
            # Adjust if nodes are crowded?
            # Simple linear spacing outward/inward
            r = base_radius + (dist_idx * spacing)
            
            # Ensure we don't go too close to center (r < 0.5)
            if r < 0.8:
                # Push outward if too close to center
                # Re-calculate spacing for inner nodes
                # If vanilla is at 4.0, and we have 5 nodes inside, they would be at 3.2, 2.4, 1.6, 0.8, 0.0
                # If we have more, we need to squeeze or curve
                r = max(0.8, r)
            
            # Convert polar to cartesian
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            
            # Check for overlap with already placed nodes (simple check)
            # If overlap, shift slightly tangentially? Or radially?
            # For now, just trust the radial spacing
            
            pos[n] = (x, y)
    
    # Fallback for any missed nodes
    for n in graph.nodes():
        if n not in pos:
            pos[n] = (0, 0) # Center
            
    print("Sacred Ring layout calculated.", flush=True)

    # --- VISUALIZATION IMPROVEMENTS ---
    
    # helper: Prettify labels (Unicode)
    def prettify_roman_numeral(label):
        """Convert standard roman numeral strings to pretty unicode (superscripts/subscripts)."""
        if label == "misc" or label == "[MISC]":
            return "Misc"
            
        # Map for superscripts
        supers = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', 
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            '+': '⁺', '-': '⁻', 'o': '°', 'dim': '°', 'ø': 'ø'
        }
        
        # Replace 'dim' or 'o' with degree symbol
        label = label.replace('dim', '°').replace('o', '°')
        
        # Identify the numeric/symbolic suffix part
        import re
        match = re.match(r'^([a-zA-Z/#]+)(.*)$', label)
        if match:
            root, suffix = match.groups()
            
            # Special handling for inversions
            if suffix == "64":
                new_suffix = "⁶₄"
            elif suffix == "43":
                 new_suffix = "⁴₃"
            elif suffix == "42":
                 new_suffix = "⁴₂"
            else:
                # Convert all chars in suffix to superscripts if available
                new_suffix = ""
                for char in suffix:
                    new_suffix += supers.get(char, char)
            
            return root + new_suffix
        return label

    # Apply pretty labels
    display_labels = {n: prettify_roman_numeral(str(n)) for n in graph.nodes()}

    # Force Aspect Ratio to Equal so circle is a circle
    ax.set_aspect('equal')
    
    # Scale graph to fit nicely
    max_r = 0
    for p in pos.values():
        r = (p[0]**2 + p[1]**2)**0.5
        if r > max_r:
            max_r = r
    
    limit = max(max_r * 1.2, base_radius * 1.5) # ensure enough space
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    
    # Draw concentric circles for visual reference
    radial_spacing = 2.0  # Updated to match increased node spacing
    import matplotlib.patches as mpatches
    
    # Dark gray circles inside and outside vanilla ring (base_radius)
    dark_gray_circle_inside = mpatches.Circle((0, 0), base_radius - radial_spacing, 
                                               fill=False, edgecolor='darkgray', 
                                               linewidth=1.5, linestyle='-', alpha=0.6)
    dark_gray_circle_outside = mpatches.Circle((0, 0), base_radius + radial_spacing, 
                                                fill=False, edgecolor='darkgray', 
                                                linewidth=1.5, linestyle='-', alpha=0.6)
    ax.add_patch(dark_gray_circle_inside)
    ax.add_patch(dark_gray_circle_outside)
    
    # Light gray circles at other modification levels
    # Draw circles at intervals of radial_spacing, up to a reasonable range
    max_levels = 4  # Draw up to 4 levels inward and outward
    for level in range(2, max_levels + 1):
        # Inward levels
        r_inward = base_radius - (level * radial_spacing)
        if r_inward > 0.5:  # Don't draw if too close to center
            light_circle_in = mpatches.Circle((0, 0), r_inward, 
                                             fill=False, edgecolor='lightgray', 
                                             linewidth=1, linestyle='--', alpha=0.4)
            ax.add_patch(light_circle_in)
        
        # Outward levels
        r_outward = base_radius + (level * radial_spacing)
        light_circle_out = mpatches.Circle((0, 0), r_outward, 
                                           fill=False, edgecolor='lightgray', 
                                           linewidth=1, linestyle='--', alpha=0.4)
        ax.add_patch(light_circle_out)
    
    # Draw static elements (edges and labels)
    edges = graph.edges()
    edge_weights = [graph[u][v].get('weight', 1) for u, v in edges]
    max_weight = max(edge_weights) if edge_weights else 1
    
    # Draw edges with varying widths based on weight
    # Use curved edges (connectionstyle='arc3,rad=0.1') to reduce visual clutter from crossings
    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        width=[w * 2 / max_weight + 0.5 for w in edge_weights],
        alpha=0.4,
        edge_color='gray',
        arrows=True,
        arrowsize=15,
        arrowstyle='->',
        connectionstyle='arc3,rad=0.1'  # Slightly curved edges to reduce visual clutter
    )
    
    # Draw edge labels (transition counts)
    edge_labels = {(u, v): graph[u][v].get('weight', 1) for u, v in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax, font_size=7, alpha=0.7)
    
    # Draw nodes
    # Initialize node colors (all light blue initially)
    node_colors = ['lightblue'] * len(graph.nodes())
    node_list = list(graph.nodes())
    
    # Draw initial nodes (Larger size with increased spacing to prevent overlap)
    nodes_drawn = nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=node_colors,
        node_size=900, # Increased size with larger spacing to prevent overlap
        alpha=0.9,
        edgecolors='white',
        linewidths=2
    )
    
    # Draw pretty labels using display_labels (increased font size for larger nodes)
    nx.draw_networkx_labels(graph, pos, labels=display_labels, ax=ax, font_size=10, font_weight='bold', font_family='DejaVu Sans')

    # RESIZING HANDLER
    def on_resize(event):
        """Adjust layout scaling on window resize"""
        pass

    fig.canvas.mpl_connect('resize_event', on_resize)
    
    unique_chord_count = len(set(chord_labels))
    ax.set_title(f"Chord Transition Diagram - Real-time Playback ({unique_chord_count} unique chords)", fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Save static version
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Real-time playback variables
    current_chord_index = [0]  # Use list to allow modification in nested functions
    previous_chord_index = [-1]  # Track previous chord index to only update on change (start at -1 so first chord is drawn)
    previous_chord_label = [None]  # Track previous chord label (string) to only update when chord actually changes
    playback_active = [True]
    animation_obj = [None]  # Store animation object to stop it
    
    # Create "End Playback" button
    from matplotlib.widgets import Button
    
    # Adjust layout to make room for button
    plt.subplots_adjust(bottom=0.1)
    
    # Create button axes (position: left, bottom, width, height in figure coordinates)
    button_ax = plt.axes([0.85, 0.02, 0.12, 0.04])
    stop_button = Button(button_ax, 'End Playback', color='lightcoral', hovercolor='red')
    
    def stop_playback(event):
        """Stop playback and animation when button is clicked"""
        print("End Playback button clicked. Stopping playback...", flush=True)
        playback_active[0] = False
        if animation_obj[0] is not None:
            animation_obj[0].event_source.stop()
        # Close the window
        plt.close(fig)
    
    stop_button.on_clicked(stop_playback)
    
    def update_visualization(frame):
        """Update the visualization to highlight the current chord."""
        if not playback_active[0] or current_chord_index[0] >= len(chord_labels):
            return nodes_drawn
        
        # Get current chord label
        current_label = chord_labels[current_chord_index[0]]
        
        # Only update if the chord index OR label string has changed
        if current_chord_index[0] == previous_chord_index[0] and current_label == previous_chord_label[0]:
            return nodes_drawn
        
        # Update previous chord index and label
        previous_chord_index[0] = current_chord_index[0]
        previous_chord_label[0] = current_label
        
        # Update node colors
        new_colors = []
        for node in node_list:
            if node == current_label:
                new_colors.append('red')  # Highlight current chord in red
            else:
                new_colors.append('lightblue')
        
        # Update the nodes
        nodes_drawn.set_color(new_colors)
        
        # Update title with current chord (pretty)
        pretty_current = prettify_roman_numeral(current_label)
        
        unique_chord_count = len(set(chord_labels))
        ax.set_title(
            f"Chord Transition Diagram - Currently Playing: {pretty_current} ({current_chord_index[0] + 1}/{len(chord_labels)} chords, {unique_chord_count} unique)",
            fontsize=16,
            fontweight='bold'
        )
        
        return nodes_drawn
    
    def play_midi():
        """
        Play MIDI via Python-only pre-rendered audio synthesis for clean, smooth playback.
        
        NOTE: A standalone, modular version of this function is available in:
        midi_audio_synthesis.py - Use that file for development and testing!
        """
        try:
            # Get tempo from MIDI file (default to 120 BPM)
            tempo = 120  # beats per minute
            try:
                midi_stream_temp = converter.parse(midi_file_path)
                tempo_marks = midi_stream_temp.flatten().getElementsByClass('TempoIndication')
                if tempo_marks:
                    tempo = tempo_marks[0].getQuarterBPM()
            except:
                pass
            
            seconds_per_quarter = 60.0 / tempo
            
            # Convert chord offsets and durations to seconds
            chord_times_seconds = [offset * seconds_per_quarter for offset in chord_offsets]
            
            # Require sounddevice and numpy for Python-only playback
            try:
                import mido
                import numpy as np
                import sounddevice as sd
            except ImportError as e:
                print(f"ERROR: Required libraries not installed: {e}", flush=True)
                print("Install with: pip install mido numpy sounddevice", flush=True)
                print("The visualization will continue with timing simulation.", flush=True)
                playback_start_time = time.time()
                # Update chord index based on simulated time
                while playback_active[0] and current_chord_index[0] < len(chord_labels) - 1:
                    elapsed = time.time() - playback_start_time
                    new_index = 0
                    for i, chord_time in enumerate(chord_times_seconds):
                        if elapsed >= chord_time:
                            new_index = i
                        else:
                            break
                    if new_index < len(chord_labels):
                        current_chord_index[0] = min(new_index, len(chord_labels) - 1)
                    time.sleep(0.05)
                playback_active[0] = False
                return
            
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
            # Use mido's merge_tracks to combine all tracks
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
                        # Add release time
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
            
            # Handle any remaining active notes (notes that never got note_off)
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
            playback_start_time = time.time()
            
            def play_audio():
                """Play pre-rendered audio buffer"""
                try:
                    sd.play(audio_buffer, samplerate=SAMPLE_RATE, blocking=False)
                    sd.wait()  # Wait for playback to finish
                except Exception as e:
                    print(f"Error during audio playback: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                finally:
                    playback_active[0] = False
            
            # Start playback in a separate thread
            playback_thread = threading.Thread(target=play_audio, daemon=True)
            playback_thread.start()
            print("MIDI playback started via pre-rendered audio. Audio should be playing cleanly now.", flush=True)
            
            # Update chord index based on actual time, synchronized with playback
            while playback_active[0] and current_chord_index[0] < len(chord_labels) - 1:
                elapsed = time.time() - playback_start_time
                
                # Find the current chord based on elapsed time
                new_index = 0
                for i, chord_time in enumerate(chord_times_seconds):
                    if elapsed >= chord_time:
                        new_index = i
                    else:
                        break
                
                # Clamp to valid range
                if new_index < len(chord_labels):
                    current_chord_index[0] = min(new_index, len(chord_labels) - 1)
                
                time.sleep(0.05)  # Update 20 times per second for smoother animation
            
            playback_active[0] = False
        except Exception as e:
            print(f"Error in MIDI playback: {e}", flush=True)
            import traceback
            traceback.print_exc()
            playback_active[0] = False
    
    # Start MIDI playback in a separate thread
    playback_thread = threading.Thread(target=play_midi, daemon=True)
    playback_thread.start()
    
    # Show the window FIRST before creating animation
    print("Opening visualization window...", flush=True)
    plt.ion()  # Turn on interactive mode
    
    # Ensure window doesn't stay on top (normal window behavior)
    try:
        # For TkAgg backend, set window attributes
        if hasattr(fig.canvas, 'manager') and hasattr(fig.canvas.manager, 'window'):
            window = fig.canvas.manager.window
            if hasattr(window, 'attributes'):
                # On Windows, ensure window doesn't stay on top
                try:
                    window.attributes('-topmost', False)
                except:
                    pass
    except:
        pass
    
    plt.show(block=False)  # Show window non-blocking first
    
    # Give the window a moment to appear
    import time
    time.sleep(0.5)
    
    # Force a draw
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        print("Window drawn and flushed.", flush=True)
    except Exception as e:
        print(f"Warning drawing window: {e}", flush=True)
    
    # Create animation AFTER window is shown
    print("Creating animation...", flush=True)
    try:
        ani = animation.FuncAnimation(
            fig, update_visualization, interval=50, blit=False, cache_frame_data=False, repeat=True
        )
        animation_obj[0] = ani  # Store for button to stop
        print("Animation created successfully.", flush=True)
    except Exception as e:
        print(f"ERROR creating animation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        ani = None
        animation_obj[0] = None
    
    print("=" * 70, flush=True)
    print("WINDOW SHOULD BE VISIBLE NOW", flush=True)
    print("Close the window to stop playback.", flush=True)
    print("=" * 70, flush=True)
    
    # Keep window open and responsive
    try:
        while plt.get_fignums():
            plt.pause(0.1)  # This keeps the window responsive and processes events
            if not playback_active[0]:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
    
    print("Visualization window closed.", flush=True)
    
    return fig, ani


def main(midi_file_path: str, output_graph_path: str = "chord_transition_graph.png"):
    """
    Main function that orchestrates the entire analysis pipeline.
    
    Args:
        midi_file_path: Path to the input MIDI file
        output_graph_path: Path to save the output graph visualization
    """
    print(f"Opening MIDI file: {midi_file_path}")
    midi_stream = open_midi_file(midi_file_path)
    
    print("Determining key signature...")
    key_sig = determine_key_signature(midi_stream)
    print(f"Detected key: {key_sig}")
    
    print("Extracting chords...")
    chords, chord_offsets, chord_durations = extract_chords(midi_stream)
    print(f"Found {len(chords)} chords")
    unique_chords = len(set(chords))
    print(f"Unique chords: {unique_chords}")
    
    print("Determining chord transitions...")
    transitions = determine_chord_transitions(chords)
    print(f"Found {len(transitions)} transitions")
    
    print("Normalizing to roman numerals...")
    
    # Debug: Show some example chords before conversion
    print(f"\nSample chords before roman numeral conversion:")
    for i, ch in enumerate(chords[:5]):
        pitch_names = [p.name for p in ch.pitches[:4]]  # Show first 4 pitches
        print(f"  Chord {i}: {', '.join(pitch_names)} (root: {ch.root().name})")
    
    roman_numerals = normalize_to_roman_numerals(chords, key_sig)
    print(f"Roman numerals: {roman_numerals[:10]}...")  # Show first 10
    
    # Debug: Show unique roman numerals and their counts
    from collections import Counter
    rn_counts = Counter(roman_numerals)
    print(f"\nUnique roman numerals found: {len(rn_counts)}")
    print("All roman numerals and their counts:")
    for rn, count in sorted(rn_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rn}: {count} occurrences")
    
    # Check specifically for I chord
    i_chord_variants = [rn for rn in rn_counts.keys() if rn.startswith('I') or rn == 'i']
    if i_chord_variants:
        print(f"\nI chord variants found: {i_chord_variants}")
    else:
        print(f"\nWARNING: No I chord found! Key is {key_sig}")
        print("Checking what chords map to the tonic...")
        # Find chords with root matching the key tonic
        tonic_pc = key_sig.tonic.pitchClass
        tonic_chords = []
        for i, ch in enumerate(chords[:20]):  # Check first 20
            if ch.root().pitchClass == tonic_pc:
                tonic_chords.append((i, ch, roman_numerals[i]))
        if tonic_chords:
            print(f"Found {len(tonic_chords)} chords with tonic root:")
            for idx, ch, rn in tonic_chords[:5]:
                print(f"  Position {idx}: {rn} (pitches: {[p.name for p in ch.pitches[:4]]})")
    
    print("Creating transition graph...", flush=True)
    try:
        graph, chord_to_label, chord_labels = create_transition_graph(chords, transitions, key_sig)
        print(f"Graph created with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges", flush=True)
    except Exception as e:
        print(f"ERROR creating graph: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    print(f"Visualizing graph with real-time playback...", flush=True)
    print("Close the visualization window to stop playback.", flush=True)
    try:
        visualize_graph(graph, chords, chord_labels, chord_offsets, chord_durations, midi_file_path, key_sig, output_graph_path)
        print("Analysis complete!", flush=True)
    except Exception as e:
        print(f"ERROR in visualization: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import sys
    
    if not LIBRARIES_AVAILABLE:
        print("\nERROR: Required libraries are not installed.", file=sys.stderr, flush=True)
        print("Please run: pip install music21 networkx matplotlib", file=sys.stderr, flush=True)
        sys.exit(1)
    
    # Get MIDI file path from command-line argument
    if len(sys.argv) < 2:
        print("Usage: python chord_transition_analyzer.py <midi_file_path>", file=sys.stderr, flush=True)
        print("Example: python chord_transition_analyzer.py letitbe.mid", file=sys.stderr, flush=True)
        sys.exit(1)
    
    midi_file = sys.argv[1]
    
    if not os.path.exists(midi_file):
        print(f"MIDI file not found: {midi_file}", flush=True)
        print(f"Current directory: {os.getcwd()}", flush=True)
        print("Please ensure the MIDI file exists.", flush=True)
        sys.exit(1)
    
    # Generate output filename based on input filename
    base_name = os.path.splitext(os.path.basename(midi_file))[0]
    output_graph_path = f"chord_transition_graph_{base_name}.png"
    
    print("=" * 50, flush=True)
    print("Analyzing MIDI file:", midi_file, flush=True)
    print("=" * 50, flush=True)
    
    # Run the full analysis pipeline
    main(midi_file, output_graph_path)
