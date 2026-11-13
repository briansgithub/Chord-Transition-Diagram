"""
Chord Detection Module

A standalone, modular function for detecting and extracting chords from MIDI files.
This module extracts the chord detection logic from the main analyzer so it can be
developed and tested independently.

The algorithm:
1. Chordifies multi-track MIDI to a single staff
2. Extracts simultaneous notes as chords
3. Normalizes chords by pitch class (removes octave duplicates)
4. Maps chords to allowed list from chords.csv, with unmatched chords going to "misc"
"""

import csv
import os
import re
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter

try:
    from music21 import stream, chord, converter, key, roman
    LIBRARIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: music21 library not available: {e}")
    print("Install with: pip install music21")
    LIBRARIES_AVAILABLE = False
    stream = None
    chord = None
    converter = None
    key = None
    roman = None


def load_allowed_chords_from_csv(csv_path: str = "chords.csv") -> Set[str]:
    """
    Load allowed chord names from chords.csv file.
    
    Args:
        csv_path: Path to the chords.csv file
        
    Returns:
        Set of allowed chord names (roman numerals) from the CSV
    """
    allowed_chords = set()
    
    if not os.path.exists(csv_path):
        print(f"Warning: chords.csv not found at {csv_path}, using empty allowed list")
        return allowed_chords
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header row
            
            for row in reader:
                if len(row) < 2:
                    continue
                
                # Process each cell in the row (skip first column which is scale degree)
                for cell in row[1:]:
                    if not cell:
                        continue
                    
                    # Split by semicolon to handle multiple chords in one cell
                    chord_names = [c.strip() for c in cell.split(';')]
                    for chord_name in chord_names:
                        if chord_name:
                            allowed_chords.add(chord_name)
        
        print(f"Loaded {len(allowed_chords)} allowed chords from {csv_path}")
        return allowed_chords
    except Exception as e:
        print(f"Error reading chords.csv: {e}")
        return allowed_chords


def create_misc_chord() -> chord.Chord:
    """
    Create a special "misc" chord to represent unmatched chords.
    Uses a distinctive pitch combination that's unlikely to occur naturally.
    
    Returns:
        A chord.Chord object representing "misc"
    """
    if not LIBRARIES_AVAILABLE:
        raise ImportError("music21 library is required")
    
    # Use a distinctive combination: C, D#, F#, A (augmented-like but distinctive)
    from music21 import pitch
    misc_pitches = [pitch.Pitch('C4'), pitch.Pitch('D#4'), pitch.Pitch('F#4'), pitch.Pitch('A4')]
    return chord.Chord(misc_pitches)


def chord_to_roman_numeral(ch: chord.Chord, key_sig: key.Key) -> Optional[str]:
    """
    Convert a chord to roman numeral notation.
    
    Args:
        ch: music21 Chord object
        key_sig: music21 Key object
        
    Returns:
        Roman numeral string (e.g., 'I', 'ii', 'V7') or None if conversion fails
    """
    try:
        if len(ch.pitches) < 2:
            # Single note - try to create a triad
            root_pitch = ch.pitches[0]
            triad = chord.Chord([root_pitch, root_pitch.transpose(4), root_pitch.transpose(7)])
            rn = roman.romanNumeralFromChord(triad, key_sig)
        else:
            rn = roman.romanNumeralFromChord(ch, key_sig)
        return str(rn.figure)
    except Exception:
        return None


def extract_chords(
    midi_stream: stream.Stream,
    key_sig: Optional[key.Key] = None,
    csv_path: str = "chords.csv",
    use_csv_filter: bool = True
) -> Tuple[List[chord.Chord], List[float], List[float]]:
    """
    Extracts all chords from the MIDI stream with timing information.
    Groups simultaneous notes into chords and maps them to allowed chords from chords.csv.
    Chords not in the CSV are mapped to a "misc" category.
    
    Args:
        midi_stream: music21 Stream object (can be created with converter.parse(midi_file))
        key_sig: Optional music21 Key object. If None, will attempt to detect from MIDI stream.
        csv_path: Path to chords.csv file containing allowed chord names
        use_csv_filter: If True, filters chords using CSV list. If False, returns all detected chords.
        
    Returns:
        Tuple of (chords, offsets, durations) where:
        - chords: List of chord.Chord objects representing all chords in the song
        - offsets: List of float offsets (in quarter notes) for when each chord starts
        - durations: List of float durations (in quarter notes) for how long each chord lasts
    """
    if not LIBRARIES_AVAILABLE:
        raise ImportError("music21 library is required for chord detection")
    
    # Detect key signature if not provided
    if key_sig is None:
        try:
            key_objects = midi_stream.flatten().getElementsByClass(key.Key)
            if key_objects:
                key_sig = key_objects[0]
            else:
                key_signatures = midi_stream.flatten().getElementsByClass(key.KeySignature)
                if key_signatures:
                    key_sig = key_signatures[0].asKey()
                else:
                    key_sig = midi_stream.analyze('key')
        except Exception:
            print("Warning: Could not detect key signature, defaulting to C major")
            key_sig = key.Key('C')
    
    print(f"Using key signature: {key_sig}")
    
    # Load allowed chords from CSV if filtering is enabled
    allowed_chords = set()
    misc_chord = None
    if use_csv_filter:
        allowed_chords = load_allowed_chords_from_csv(csv_path)
        misc_chord = create_misc_chord()
        print(f"Allowed chords: {sorted(allowed_chords)}")
    
    # Use chordify() to reduce multi-part/track MIDI to a single staff
    # This groups simultaneous notes across tracks into chords
    chordified = midi_stream.chordify()
    
    # Extract all Chord objects from the chordified stream with their timing
    raw_chords = []
    raw_offsets = []
    raw_durations = []
    
    for ch in chordified.flatten().getElementsByClass(chord.Chord):
        raw_chords.append(ch)
        raw_offsets.append(ch.offset)
        raw_durations.append(ch.duration.quarterLength)
    
    print(f"Found {len(raw_chords)} raw chords from MIDI file")
    
    # Process chords: normalize by pitch class (remove octave duplicates)
    # This preserves chord quality while removing octave information
    processed_chords = []
    processed_offsets = []
    processed_durations = []
    
    # Track unique chord signatures to reduce duplicates
    chord_signatures: Dict[Tuple[int, ...], chord.Chord] = {}  # Maps pitch class tuple to representative chord
    
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
    
    unique_chords = list(set(processed_chords))
    print(f"After normalization: {len(unique_chords)} unique chords")
    
    # If CSV filtering is enabled, map chords to allowed list or misc
    if use_csv_filter and allowed_chords:
        final_chords = []
        final_offsets = []
        final_durations = []
        
        # --- TEMPLATE MATCHING ALGORITHM ---
        
        class ChordTemplate:
            def __init__(self, label: str, key_obj: key.Key):
                self.label = label
                try:
                    # Parse Roman Numeral
                    rn = roman.RomanNumeral(label, key_obj)
                    self.root_pc = rn.root().pitchClass
                    self.bass_pc = rn.bass().pitchClass
                    self.pitches = set(p.pitchClass for p in rn.pitches)
                    self.valid = True
                except Exception:
                    self.valid = False

        # 1. Generate Templates based on Key and CSV
        print(f"Generating chord templates for key: {key_sig}")
        templates = []
        for label in allowed_chords:
            tmpl = ChordTemplate(label, key_sig)
            if tmpl.valid:
                templates.append(tmpl)
        
        print(f"Generated {len(templates)} valid templates")
        
        # 2. Matching Function
        def find_best_match(target_chord: chord.Chord) -> chord.Chord:
            if not templates:
                return misc_chord
                
            obs_pitches = set(p.pitchClass for p in target_chord.pitches)
            if not obs_pitches:
                return misc_chord
                
            obs_bass = target_chord.bass().pitchClass
            obs_root = target_chord.root().pitchClass
            
            best_score = -1.0
            best_template = None
            
            for tmpl in templates:
                # SCORING HEURISTIC
                
                # A. Coverage: How many of the template's notes are present?
                # We care that the ESSENTIAL notes of the template are there.
                common_notes = obs_pitches.intersection(tmpl.pitches)
                if not tmpl.pitches:
                    continue
                coverage = len(common_notes) / len(tmpl.pitches)
                
                # B. Noise Penalty: Penalize notes in observation that aren't in template
                # (e.g. matching C-E-G-A to C-E-G should be penalized slightly vs C-E-G-A to Am7)
                extra_notes = len(obs_pitches - tmpl.pitches)
                noise_penalty = extra_notes * 0.1
                
                # C. Bass Bonus: Critical for inversions (I64 vs V)
                bass_bonus = 0.2 if obs_bass == tmpl.bass_pc else 0.0
                
                # D. Root Bonus: If music21 agrees on the root, give a bump
                root_bonus = 0.1 if obs_root == tmpl.root_pc else 0.0
                
                # Calculate final score
                score = coverage - noise_penalty + bass_bonus + root_bonus
                
                if score > best_score:
                    best_score = score
                    best_template = tmpl
            
            # Threshold: match must be reasonable (e.g. > 50% coverage + context)
            # A perfect triad match (3/3) = 1.0 + 0.2(bass) + 0.1(root) = 1.3
            # A partial match (2/3) = 0.66 + ...
            if best_score > 0.6 and best_template:
                # Validate that the template label is actually in allowed_chords
                if best_template.label not in allowed_chords:
                    print(f"WARNING: Template label '{best_template.label}' not in allowed_chords, mapping to misc", flush=True)
                    return misc_chord
                
                # Return a new chord object named after the template
                try:
                    rn = roman.RomanNumeral(best_template.label, key_sig)
                    # Verify the generated figure is reasonable
                    generated_figure = str(rn.figure)
                    
                    # Check if generated figure is too weird (contains ## or unusual patterns)
                    if '##' in generated_figure or generated_figure.count('#') > 2 or generated_figure.count('b') > 2:
                        print(f"WARNING: Generated figure '{generated_figure}' from template '{best_template.label}' looks invalid, mapping to misc", flush=True)
                        return misc_chord
                    
                    # Use the pitches from the RomanNumeral
                    matched_chord = chord.Chord(rn.pitches)
                    # Store the ORIGINAL template label, not the generated figure
                    matched_chord.roman_label = best_template.label
                    return matched_chord
                except Exception as e:
                    print(f"WARNING: Error creating RomanNumeral from '{best_template.label}': {e}, mapping to misc", flush=True)
                    return misc_chord
            
            return misc_chord

        # 3. Process all chords
        misc_count = 0
        chord_to_final_map = {} # Cache for unique chords
        
        for i, ch in enumerate(processed_chords):
            # Use cached result if available (processed_chords has duplicates, but objects might differ)
            # We used 'unique_chords' earlier for logic, but here we iterate the full list.
            # Let's map based on pitch signature to speed up.
            
            # Optimization: Map pitch tuple to result
            pitch_sig = tuple(sorted(set(p.pitchClass for p in ch.pitches)) + [ch.bass().pitchClass])
            
            if pitch_sig in chord_to_final_map:
                mapped_chord = chord_to_final_map[pitch_sig]
            else:
                mapped_chord = find_best_match(ch)
                chord_to_final_map[pitch_sig] = mapped_chord
            
            # Final validation: Check if the roman_label (if set) is actually in allowed_chords
            if hasattr(mapped_chord, 'roman_label'):
                label = mapped_chord.roman_label
                # Check if label is in allowed_chords (exact match or normalized)
                label_normalized = label.replace('°', '').replace('dim', '').replace('o', '')
                is_valid = (label in allowed_chords or 
                           label_normalized in allowed_chords or
                           any(label.startswith(allowed) or allowed.startswith(label) 
                               for allowed in allowed_chords))
                
                if not is_valid:
                    # Invalid label detected - remap to misc
                    print(f"WARNING: Label '{label}' not in allowed_chords, remapping to misc", flush=True)
                    mapped_chord = misc_chord
            
            final_chords.append(mapped_chord)
            
            # Check if misc (by comparing pitches to misc_chord)
            if set(p.pitchClass for p in mapped_chord.pitches) == set(p.pitchClass for p in misc_chord.pitches):
                misc_count += 1
            
            final_offsets.append(processed_offsets[i])
            final_durations.append(processed_durations[i])
        
        # Print statistics
        allowed_count = len(final_chords) - misc_count
        print(f"Final chord count: {len(set(final_chords))} unique chords")
        print(f"  - {allowed_count} chords matched allowed list")
        print(f"  - {misc_count} chords mapped to 'misc' category")
        
        return final_chords, final_offsets, final_durations
    
    print(f"Final chord count: {len(unique_chords)} unique chords")
    return processed_chords, processed_offsets, processed_durations


def analyze_chord_statistics(chords: List[chord.Chord]) -> Dict:
    """
    Analyze statistics about detected chords.
    
    Args:
        chords: List of chord.Chord objects
        
    Returns:
        Dictionary with statistics:
        - total_chords: Total number of chords
        - unique_chords: Number of unique chords
        - chord_counts: Counter of chord frequencies
        - chord_types: List of unique chord types
        - average_notes_per_chord: Average number of notes in chords
    """
    chord_counts = Counter(chords)
    unique_chords = list(set(chords))
    
    total_notes = sum(len(ch.pitches) for ch in chords)
    avg_notes = total_notes / len(chords) if chords else 0
    
    return {
        'total_chords': len(chords),
        'unique_chords': len(unique_chords),
        'chord_counts': chord_counts,
        'chord_types': unique_chords,
        'average_notes_per_chord': avg_notes
    }


def print_chord_info(chords: List[chord.Chord], offsets: List[float], durations: List[float], max_display: int = 20, misc_chord: Optional[chord.Chord] = None):
    """
    Print information about detected chords for debugging.
    
    Args:
        chords: List of chord.Chord objects
        offsets: List of chord start times
        durations: List of chord durations
        max_display: Maximum number of chords to display
        misc_chord: Optional misc chord object to identify and label specially
    """
    print("\n" + "=" * 70)
    print("CHORD DETECTION RESULTS")
    print("=" * 70)
    print(f"Total chords detected: {len(chords)}")
    print(f"Unique chords: {len(set(chords))}")
    print(f"\nFirst {min(max_display, len(chords))} chords:")
    print("-" * 70)
    
    # Identify misc chord if provided
    misc_pitch_classes = None
    if misc_chord:
        misc_pitch_classes = tuple(sorted(set([p.pitchClass for p in misc_chord.pitches])))
    
    for i in range(min(max_display, len(chords))):
        ch = chords[i]
        pitch_names = [p.name for p in ch.pitches]
        root_name = ch.root().name
        
        # Check if this is the misc chord
        ch_pitch_classes = tuple(sorted(set([p.pitchClass for p in ch.pitches])))
        is_misc = (misc_pitch_classes is not None and ch_pitch_classes == misc_pitch_classes)
        
        label_suffix = ""
        if is_misc:
            label_suffix = " [MISC]"
        elif hasattr(ch, 'roman_label'):
             label_suffix = f" [{ch.roman_label}]"
        
        print(f"Chord {i+1:3d}: {', '.join(pitch_names):20s} | Root: {root_name:3s} | "
              f"Offset: {offsets[i]:6.2f} | Duration: {durations[i]:6.2f}{label_suffix}")
    
    if len(chords) > max_display:
        print(f"\n... and {len(chords) - max_display} more chords")
    
    # Show unique chord types
    unique_chords = list(set(chords))
    print(f"\nUnique chord types ({len(unique_chords)} total):")
    print("-" * 70)
    chord_counts = Counter(chords)
    for ch, count in chord_counts.most_common(10):
        pitch_names = [p.name for p in ch.pitches]
        root_name = ch.root().name
        
        # Check if this is the misc chord
        ch_pitch_classes = tuple(sorted(set([p.pitchClass for p in ch.pitches])))
        is_misc = (misc_pitch_classes is not None and ch_pitch_classes == misc_pitch_classes)
        
        label_suffix = ""
        if is_misc:
            label_suffix = " [MISC]"
        elif hasattr(ch, 'roman_label'):
             label_suffix = f" [{ch.roman_label}]"
        
        print(f"  {', '.join(pitch_names):20s} (root: {root_name:3s}) - appears {count} times{label_suffix}")
    
    print("=" * 70 + "\n")


# Test harness
if __name__ == "__main__":
    import os
    import sys
    
    # Test on maple.mid if available
    midi_file = "maple.mid"
    
    if len(sys.argv) > 1:
        midi_file = sys.argv[1]
    
    if not os.path.exists(midi_file):
        print(f"MIDI file not found: {midi_file}")
        print(f"Current directory: {os.getcwd()}")
        print("Usage: python chord_detection.py [midi_file_path]")
        sys.exit(1)
    
    print("=" * 70)
    print("Testing Chord Detection Algorithm")
    print("=" * 70)
    print(f"MIDI file: {midi_file}\n")
    
    try:
        # Load MIDI file
        print("Loading MIDI file...")
        midi_stream = converter.parse(midi_file)
        print("MIDI file loaded successfully.\n")
        
        # Extract chords
        print("Extracting chords...")
        chords, offsets, durations = extract_chords(midi_stream, key_sig=None, csv_path="chords.csv", use_csv_filter=True)
        
        # Create misc chord for display purposes
        misc_chord = create_misc_chord()
        
        # Print results
        print_chord_info(chords, offsets, durations, max_display=30, misc_chord=misc_chord)
        
        # Analyze statistics
        stats = analyze_chord_statistics(chords)
        print("Statistics:")
        print(f"  Total chords: {stats['total_chords']}")
        print(f"  Unique chords: {stats['unique_chords']}")
        print(f"  Average notes per chord: {stats['average_notes_per_chord']:.2f}")
        
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

