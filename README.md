# Chordography (Chord Transition Diagram)

A Python tool that analyzes a MIDI file, estimates its key, extracts chords, and draws a Markov-style graph of chord transitions in roman-numeral notation.

## Usage

Install dependencies and run the analyzer:

```
pip install -r requirements.txt
python chord_transition_analyzer.py
```

Place MIDI files in the `midis/` folder. Example graphs for Maple Leaf Rag and Let It Be are included. A large optional Chordonomicon CSV, if present locally, is gitignored and is not part of this repository.
