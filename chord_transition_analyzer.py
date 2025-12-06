"""
Chord Transition Diagram Generator

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
from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict

# Required libraries (install with: pip install music21 networkx matplotlib)
try:
    from music21 import stream, midi, key, chord, roman, analysis, converter
    import networkx as nx
    # Set matplotlib backend BEFORE importing pyplot
    import matplotlib
    matplotlib.use('TkAgg')  # Force TkAgg for Windows compatibility
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    LIBRARIES_AVAILABLE = True
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


def extract_chords(midi_stream: stream.Stream) -> Tuple[List[chord.Chord], List[float], List[float]]:
    """
    Extracts all chords from the MIDI stream with timing information.
    Groups simultaneous notes into chords and simplifies them to at most 25 unique chords.
    Chords are normalized to root position and simplified to standard triads and 7th chords.
    
    Args:
        midi_stream: music21 Stream object
        
    Returns:
        Tuple of (chords, offsets, durations) where:
        - chords: List of chord.Chord objects representing all chords in the song (simplified)
        - offsets: List of float offsets (in quarter notes) for when each chord starts
        - durations: List of float durations (in quarter notes) for how long each chord lasts
    """
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
    
    # Simplify chords: normalize to root position and reduce to essential intervals
    simplified_chords = []
    simplified_offsets = []
    simplified_durations = []
    chord_signatures = {}  # Maps chord signature to simplified chord
    
    for i, ch in enumerate(raw_chords):
        # Get the root note
        root_note = ch.root()
        root_pitch_class = root_note.pitchClass
        
        # Get unique pitch classes (remove octave information)
        pitch_classes = sorted(set([p.pitchClass for p in ch.pitches]))
        
        # Normalize to root position (shift so root is at 0)
        normalized_pcs = [(pc - root_pitch_class) % 12 for pc in pitch_classes]
        normalized_pcs.sort()
        
        # Simplify to standard chord types (triads and 7ths)
        # Keep only essential intervals: root (0), 3rd (3 or 4), 5th (7), 7th (10 or 11)
        essential_intervals = [0]  # Always include root
        
        # Add 3rd if present (major 3rd = 4, minor 3rd = 3)
        if 3 in normalized_pcs or 4 in normalized_pcs:
            # Prefer minor 3rd if both present, otherwise use what's there
            if 3 in normalized_pcs:
                essential_intervals.append(3)
            elif 4 in normalized_pcs:
                essential_intervals.append(4)
        
        # Add 5th if present (perfect 5th = 7, diminished 5th = 6, augmented 5th = 8)
        if 7 in normalized_pcs:
            essential_intervals.append(7)
        elif 6 in normalized_pcs:
            essential_intervals.append(6)  # Diminished 5th
        elif 8 in normalized_pcs:
            essential_intervals.append(8)  # Augmented 5th
        
        # Add 7th if present (minor 7th = 10, major 7th = 11)
        if 10 in normalized_pcs or 11 in normalized_pcs:
            if 10 in normalized_pcs:
                essential_intervals.append(10)  # Minor 7th
            elif 11 in normalized_pcs:
                essential_intervals.append(11)  # Major 7th
        
        # If we only have root, try to infer a chord from context
        # For single notes, create a simple triad based on the key context
        if len(essential_intervals) == 1:
            # Single note - create a major triad (most common)
            essential_intervals = [0, 4, 7]
        
        # Create a signature from the normalized intervals
        chord_signature = tuple(sorted(essential_intervals))
        
        # If we've seen this chord type before, reuse it
        if chord_signature in chord_signatures:
            simplified_chords.append(chord_signatures[chord_signature])
        else:
            # Create a simplified chord in root position
            simplified_pitches = []
            for interval in sorted(essential_intervals):
                new_pitch = root_note.transpose(interval)
                simplified_pitches.append(new_pitch)
            
            # Create new simplified chord
            simplified_chord = chord.Chord(simplified_pitches)
            
            # Store for reuse
            chord_signatures[chord_signature] = simplified_chord
            simplified_chords.append(simplified_chord)
        
        # Store timing information
        simplified_offsets.append(raw_offsets[i])
        simplified_durations.append(raw_durations[i])
    
    # If we have more than 25 unique chords, reduce further by keeping only the most common
    unique_chords = list(set(simplified_chords))
    
    if len(unique_chords) > 25:
        # Count frequency of each unique chord
        chord_counts = Counter(simplified_chords)
        
        # Keep the 25 most common chords
        top_25_chords = [ch for ch, _ in chord_counts.most_common(25)]
        
        # Map all chords to the closest match in top 25
        final_chords = []
        final_offsets = []
        final_durations = []
        
        for i, ch in enumerate(simplified_chords):
            if ch in top_25_chords:
                final_chords.append(ch)
            else:
                # Find the closest match (same root, similar intervals)
                root_pc = ch.root().pitchClass
                matching_chords = [c for c in top_25_chords if c.root().pitchClass == root_pc]
                if matching_chords:
                    # Use the most common one with same root
                    final_chords.append(max(matching_chords, key=lambda x: chord_counts[x]))
                else:
                    # Use the most common chord overall
                    final_chords.append(top_25_chords[0])
            
            # Preserve timing information
            final_offsets.append(simplified_offsets[i])
            final_durations.append(simplified_durations[i])
        
        return final_chords, final_offsets, final_durations
    
    return simplified_chords, simplified_offsets, simplified_durations


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
    for ch in chords:
        try:
            rn = roman.romanNumeralFromChord(ch, key_signature)
            roman_numerals.append(str(rn.figure))
        except Exception:
            # If roman numeral conversion fails, use chord name as fallback
            chord_name = ', '.join(sorted(set(ch.pitchNames)))
            roman_numerals.append(chord_name)
    
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
    
    # Convert chords to labels (roman numerals or chord names)
    chord_to_label = {}
    chord_labels = []
    
    for ch in chords:
        # Create a unique identifier for the chord
        chord_key = tuple(sorted(set([p.pitchClass for p in ch.pitches])))
        
        if chord_key not in chord_to_label:
            try:
                rn = roman.romanNumeralFromChord(ch, key_signature)
                label = str(rn.figure)
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
        fig, ax = plt.subplots(figsize=(16, 12))
        print("Figure created successfully.", flush=True)
    except Exception as e:
        print(f"ERROR creating figure: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    # Calculate layout (positions of nodes)
    pos = nx.spring_layout(graph, k=3, iterations=50, seed=42)
    
    # Draw static elements (edges and labels)
    edges = graph.edges()
    edge_weights = [graph[u][v].get('weight', 1) for u, v in edges]
    max_weight = max(edge_weights) if edge_weights else 1
    
    # Draw edges with varying widths based on weight
    nx.draw_networkx_edges(
        graph, pos, ax=ax,
        width=[w * 2 / max_weight + 0.5 for w in edge_weights],
        alpha=0.4,
        edge_color='gray',
        arrows=True,
        arrowsize=15,
        arrowstyle='->'
    )
    
    # Draw edge labels (transition counts)
    edge_labels = {(u, v): graph[u][v].get('weight', 1) for u, v in edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, ax=ax, font_size=7, alpha=0.7)
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9, font_weight='bold')
    
    # Initialize node colors (all light blue initially)
    node_colors = ['lightblue'] * len(graph.nodes())
    node_list = list(graph.nodes())
    
    # Draw initial nodes
    nodes_drawn = nx.draw_networkx_nodes(
        graph, pos, ax=ax,
        node_color=node_colors,
        node_size=3000,
        alpha=0.8
    )
    
    unique_chord_count = len(set(chord_labels))
    ax.set_title(f"Chord Transition Diagram - Real-time Playback ({unique_chord_count} unique chords)", fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # Save static version
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Real-time playback variables
    current_chord_index = [0]  # Use list to allow modification in nested functions
    playback_active = [True]
    
    def update_visualization(frame):
        """Update the visualization to highlight the current chord."""
        if not playback_active[0] or current_chord_index[0] >= len(chord_labels):
            return nodes_drawn
        
        # Get current chord label
        current_label = chord_labels[current_chord_index[0]]
        
        # Update node colors
        new_colors = []
        for node in node_list:
            if node == current_label:
                new_colors.append('red')  # Highlight current chord in red
            else:
                new_colors.append('lightblue')
        
        # Update the nodes
        nodes_drawn.set_color(new_colors)
        
        # Update title with current chord
        unique_chord_count = len(set(chord_labels))
        ax.set_title(
            f"Chord Transition Diagram - Currently Playing: {current_label} ({current_chord_index[0] + 1}/{len(chord_labels)} chords, {unique_chord_count} unique)",
            fontsize=16,
            fontweight='bold'
        )
        
        return nodes_drawn
    
    def play_midi():
        """Play MIDI via Python and update current chord index based on actual timing."""
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
            
            # Play MIDI entirely within Python using pure Python synthesizer
            print("Starting MIDI playback via Python...", flush=True)
            playback_start_time = None
            
            try:
                import mido
                import numpy as np
                
                # Read MIDI file
                mid = mido.MidiFile(midi_file_path)
                
                # Method 1: Try mido's built-in MIDI output (if backend available)
                midi_output_success = False
                try:
                    output_names = mido.get_output_names()
                    if output_names:
                        outport = mido.open_output(output_names[0])
                        print(f"Playing MIDI via mido (port: {output_names[0]})...", flush=True)
                        playback_start_time = time.time()
                        midi_output_success = True
                        
                        def play_midi_events():
                            try:
                                for msg in mid.play():
                                    if not playback_active[0]:
                                        break
                                    if not msg.is_meta:
                                        outport.send(msg)
                            except Exception as e:
                                print(f"Error during MIDI playback: {e}", flush=True)
                            finally:
                                try:
                                    outport.close()
                                except:
                                    pass
                        
                        midi_playback_thread = threading.Thread(target=play_midi_events, daemon=True)
                        midi_playback_thread.start()
                        print("MIDI playback started. Audio should be playing now.", flush=True)
                except Exception as e:
                    # MIDI output not available, try audio synthesis
                    pass
                
                # Method 2: Pure Python MIDI-to-audio synthesis using sounddevice
                if not midi_output_success:
                    try:
                        import sounddevice as sd
                        
                        # Optimized MIDI synthesizer parameters for smooth playback
                        SAMPLE_RATE = 44100
                        CHUNK = 512  # Smaller chunk for lower latency
                        
                        print("Playing MIDI via Python audio synthesis (optimized)...", flush=True)
                        playback_start_time = time.time()
                        
                        # Thread-safe note management
                        notes_lock = threading.Lock()
                        active_notes = {}  # {note: (frequency, velocity, start_time, phase_offset)}
                        
                        def midi_note_to_freq(note):
                            """Convert MIDI note number to frequency in Hz"""
                            return 440.0 * (2.0 ** ((note - 69) / 12.0))
                        
                        def audio_callback(outdata, frames, time_info, status):
                            """Optimized callback for smooth real-time audio generation"""
                            if status:
                                print(f"Audio callback status: {status}", flush=True)
                            
                            # Use stream time for better synchronization
                            stream_time = time_info['current_time']
                            elapsed = stream_time - playback_start_time
                            
                            samples = np.zeros(frames, dtype=np.float32)
                            
                            with notes_lock:
                                notes_copy = dict(active_notes)
                            
                            for note, (freq, vel, start, phase_offset) in notes_copy.items():
                                note_time = elapsed - start
                                if note_time < 0:
                                    continue
                                
                                # Generate sine wave with proper phase continuity
                                t = np.arange(frames) / SAMPLE_RATE
                                phase = 2 * np.pi * freq * (note_time + t) + phase_offset
                                wave = np.sin(phase) * (vel / 127.0) * 0.25  # Reduced volume for mixing
                                samples += wave
                            
                            # Soft clipping for smoother sound (better than hard normalization)
                            samples = np.tanh(samples * 0.8) * 0.9
                            
                            outdata[:] = samples.reshape(-1, 1)
                        
                        # Start audio stream with optimized settings
                        stream = sd.OutputStream(
                            samplerate=SAMPLE_RATE, 
                            channels=1, 
                            dtype=np.float32,
                            callback=audio_callback,
                            blocksize=CHUNK,
                            latency='low'  # Low latency for smoother playback
                        )
                        stream.start()
                        
                        def play_midi_with_synthesis():
                            """Play MIDI file with optimized audio synthesis"""
                            try:
                                # Track phase for smooth note transitions
                                note_phases = {}
                                
                                for msg in mid.play():
                                    if not playback_active[0]:
                                        break
                                    
                                    current_time = time.time() - playback_start_time
                                    
                                    if msg.type == 'note_on' and msg.velocity > 0:
                                        freq = midi_note_to_freq(msg.note)
                                        # Continue from previous phase if note was already playing
                                        phase_offset = note_phases.get(msg.note, 0.0)
                                        
                                        with notes_lock:
                                            active_notes[msg.note] = (freq, msg.velocity, current_time, phase_offset)
                                            note_phases[msg.note] = phase_offset
                                    
                                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                                        with notes_lock:
                                            if msg.note in active_notes:
                                                # Save phase for potential retrigger
                                                freq, vel, start, phase = active_notes[msg.note]
                                                note_phases[msg.note] = phase + 2 * np.pi * freq * (current_time - start)
                                                del active_notes[msg.note]
                                
                                # Shorter fade for responsiveness
                                fade_duration = 0.5
                                fade_start = time.time()
                                while active_notes and (time.time() - fade_start) < fade_duration:
                                    if not playback_active[0]:
                                        break
                                    time.sleep(0.01)
                                
                            except Exception as e:
                                print(f"Error during audio synthesis: {e}", flush=True)
                            finally:
                                try:
                                    stream.stop()
                                    stream.close()
                                except:
                                    pass
                        
                        midi_playback_thread = threading.Thread(target=play_midi_with_synthesis, daemon=True)
                        midi_playback_thread.start()
                        print("MIDI playback started via Python audio synthesis. Audio should be playing now.", flush=True)
                        
                    except ImportError:
                        # sounddevice not available, use system player with timing
                        print("sounddevice not available. Using system MIDI player with Python timing control...", flush=True)
                        print("NOTE: For smoother playback, install Visual Studio and then: pip install python-rtmidi", flush=True)
                        import subprocess
                        import platform
                        
                        playback_start_time = time.time()
                        
                        if platform.system() == 'Windows':
                            try:
                                os.startfile(midi_file_path)
                                print("MIDI file opened in Windows default player.", flush=True)
                            except:
                                subprocess.Popen(['start', midi_file_path], shell=True)
                                print("MIDI file opened via subprocess.", flush=True)
                        elif platform.system() == 'Darwin':
                            subprocess.Popen(['open', midi_file_path])
                            print("MIDI file opened in macOS default player.", flush=True)
                        else:
                            subprocess.Popen(['xdg-open', midi_file_path])
                            print("MIDI file opened in Linux default player.", flush=True)
                        
                        print("Python is controlling timing for visualization sync.", flush=True)
                        print("For best audio quality, install python-rtmidi (requires Visual Studio).", flush=True)
                    except Exception as e:
                        print(f"Audio synthesis failed: {e}", flush=True)
                        # Fall through to system player
                        import subprocess
                        import platform
                        playback_start_time = time.time()
                        if platform.system() == 'Windows':
                            try:
                                os.startfile(midi_file_path)
                            except:
                                subprocess.Popen(['start', midi_file_path], shell=True)
                        
            except ImportError as e:
                print(f"ERROR: Required libraries not installed: {e}", flush=True)
                print("Install with: pip install mido numpy pyaudio", flush=True)
                print("The visualization will continue with timing simulation.", flush=True)
                playback_start_time = time.time()
            except Exception as e:
                print(f"ERROR: MIDI playback setup failed: {e}", flush=True)
                print("The visualization will continue with timing simulation.", flush=True)
                playback_start_time = time.time()
            
            # Ensure playback_start_time is set
            if playback_start_time is None:
                playback_start_time = time.time()
            
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
            fig, update_visualization, interval=100, blit=False, cache_frame_data=False, repeat=True
        )
        print("Animation created successfully.", flush=True)
    except Exception as e:
        print(f"ERROR creating animation: {e}", flush=True)
        import traceback
        traceback.print_exc()
        ani = None
    
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
    roman_numerals = normalize_to_roman_numerals(chords, key_sig)
    print(f"Roman numerals: {roman_numerals[:10]}...")  # Show first 10
    
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
        visualize_graph(graph, chords, chord_labels, chord_offsets, chord_durations, midi_file_path, output_graph_path)
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
    
    # Test on maple.mid
    midi_file = "maple.mid"
    
    if not os.path.exists(midi_file):
        print(f"MIDI file not found: {midi_file}", flush=True)
        print(f"Current directory: {os.getcwd()}", flush=True)
        print("Please ensure the MIDI file is in the current directory.", flush=True)
        sys.exit(1)
    
    print("=" * 50, flush=True)
    print("Testing MIDI file analysis on:", midi_file, flush=True)
    print("=" * 50, flush=True)
    
    # Run the full analysis pipeline
    main(midi_file, "chord_transition_graph_maple.png")

