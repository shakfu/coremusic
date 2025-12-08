#!/usr/bin/env python3
"""Music theory primitives and utilities.

This module provides foundational music theory constructs:
- Note representation with octave and accidentals
- Intervals (semitone distances with quality)
- Scales (major, minor, modes, exotic scales)
- Chords (triads, sevenths, extended, altered)
- Key signatures and circle of fifths

All classes are designed to work seamlessly with MIDI note numbers
(0-127) while providing human-readable interfaces.

Example:
    >>> note = Note('C', 4)  # Middle C
    >>> note.midi  # 60
    >>> note.transpose(Interval.PERFECT_FIFTH)  # G4
    >>>
    >>> scale = Scale(Note('C', 4), ScaleType.MAJOR)
    >>> [n.name for n in scale.get_notes()]  # ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    >>>
    >>> chord = Chord(Note('C', 4), ChordType.MAJOR_7)
    >>> chord.get_midi_notes()  # [60, 64, 67, 71]
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Dict, Iterator, List, Optional, Tuple, Union

# ============================================================================
# Constants
# ============================================================================

# Note names in chromatic order
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Enharmonic equivalents (uppercase keys for lookup after .upper())
ENHARMONIC_MAP = {
    'DB': 'C#', 'EB': 'D#', 'FB': 'E', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#', 'CB': 'B',
    'B#': 'C', 'E#': 'F', 'C##': 'D', 'D##': 'E', 'F##': 'G', 'G##': 'A', 'A##': 'B',
    'DBB': 'C', 'EBB': 'D', 'FBB': 'E', 'GBB': 'F', 'ABB': 'G', 'BBB': 'A', 'CBB': 'B',
}

# Flat note names for display
FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Key signatures: key -> (sharps/flats list, is_minor)
KEY_SIGNATURES: Dict[str, Tuple[List[str], bool]] = {
    # Major keys
    'C': ([], False),
    'G': (['F#'], False),
    'D': (['F#', 'C#'], False),
    'A': (['F#', 'C#', 'G#'], False),
    'E': (['F#', 'C#', 'G#', 'D#'], False),
    'B': (['F#', 'C#', 'G#', 'D#', 'A#'], False),
    'F#': (['F#', 'C#', 'G#', 'D#', 'A#', 'E#'], False),
    'C#': (['F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#'], False),
    'F': (['Bb'], False),
    'Bb': (['Bb', 'Eb'], False),
    'Eb': (['Bb', 'Eb', 'Ab'], False),
    'Ab': (['Bb', 'Eb', 'Ab', 'Db'], False),
    'Db': (['Bb', 'Eb', 'Ab', 'Db', 'Gb'], False),
    'Gb': (['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb'], False),
    'Cb': (['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb', 'Fb'], False),
    # Minor keys (relative minors)
    'Am': ([], True),
    'Em': (['F#'], True),
    'Bm': (['F#', 'C#'], True),
    'F#m': (['F#', 'C#', 'G#'], True),
    'C#m': (['F#', 'C#', 'G#', 'D#'], True),
    'G#m': (['F#', 'C#', 'G#', 'D#', 'A#'], True),
    'D#m': (['F#', 'C#', 'G#', 'D#', 'A#', 'E#'], True),
    'A#m': (['F#', 'C#', 'G#', 'D#', 'A#', 'E#', 'B#'], True),
    'Dm': (['Bb'], True),
    'Gm': (['Bb', 'Eb'], True),
    'Cm': (['Bb', 'Eb', 'Ab'], True),
    'Fm': (['Bb', 'Eb', 'Ab', 'Db'], True),
    'Bbm': (['Bb', 'Eb', 'Ab', 'Db', 'Gb'], True),
    'Ebm': (['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb'], True),
    'Abm': (['Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb', 'Fb'], True),
}

# Circle of fifths (clockwise from C)
CIRCLE_OF_FIFTHS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']


# ============================================================================
# Utility Functions
# ============================================================================


def note_name_to_midi(name: str, octave: int = 4) -> int:
    """Convert note name to MIDI note number.

    Args:
        name: Note name (e.g., 'C', 'C#', 'Db', 'C4')
        octave: Octave number (default 4, middle C octave)

    Returns:
        MIDI note number (0-127)

    Raises:
        ValueError: If note name is invalid

    Example:
        >>> note_name_to_midi('C', 4)
        60
        >>> note_name_to_midi('A', 4)
        69
        >>> note_name_to_midi('C#4')
        61
    """
    # Parse octave from name if present (e.g., "C4", "F#3")
    match = re.match(r'^([A-Ga-g][#b]*)(-?\d+)?$', name)
    if not match:
        raise ValueError(f"Invalid note name: {name}")

    note_part = match.group(1).upper()
    if match.group(2) is not None:
        octave = int(match.group(2))

    # Handle enharmonic equivalents
    if note_part in ENHARMONIC_MAP:
        note_part = ENHARMONIC_MAP[note_part]

    # Find pitch class
    try:
        pitch_class = NOTE_NAMES.index(note_part)
    except ValueError:
        raise ValueError(f"Unknown note name: {note_part}")

    midi = (octave + 1) * 12 + pitch_class
    if not 0 <= midi <= 127:
        raise ValueError(f"MIDI note {midi} out of range (0-127)")

    return midi


def midi_to_note_name(midi: int, use_flats: bool = False) -> str:
    """Convert MIDI note number to note name with octave.

    Args:
        midi: MIDI note number (0-127)
        use_flats: If True, use flat names (Db) instead of sharps (C#)

    Returns:
        Note name with octave (e.g., 'C4', 'F#3')

    Example:
        >>> midi_to_note_name(60)
        'C4'
        >>> midi_to_note_name(61)
        'C#4'
        >>> midi_to_note_name(61, use_flats=True)
        'Db4'
    """
    if not 0 <= midi <= 127:
        raise ValueError(f"MIDI note {midi} out of range (0-127)")

    octave = (midi // 12) - 1
    pitch_class = midi % 12

    names = FLAT_NAMES if use_flats else NOTE_NAMES
    return f"{names[pitch_class]}{octave}"


# ============================================================================
# Interval Types
# ============================================================================


class IntervalQuality(Enum):
    """Interval quality (perfect, major, minor, augmented, diminished)."""
    PERFECT = "P"
    MAJOR = "M"
    MINOR = "m"
    AUGMENTED = "A"
    DIMINISHED = "d"


@dataclass(frozen=True)
class Interval:
    """Musical interval (distance between two notes).

    Attributes:
        semitones: Number of semitones
        quality: Interval quality
        number: Interval number (1=unison, 2=second, etc.)
        name: Human-readable name

    Example:
        >>> Interval.PERFECT_FIFTH
        Interval(semitones=7, name='Perfect Fifth')
        >>> Interval.from_semitones(7)
        Interval.PERFECT_FIFTH
    """
    semitones: int
    quality: IntervalQuality
    number: int
    name: str

    # Class-level interval constants (assigned below class definition)
    UNISON: ClassVar["Interval"]
    MINOR_SECOND: ClassVar["Interval"]
    MAJOR_SECOND: ClassVar["Interval"]
    MINOR_THIRD: ClassVar["Interval"]
    MAJOR_THIRD: ClassVar["Interval"]
    PERFECT_FOURTH: ClassVar["Interval"]
    TRITONE: ClassVar["Interval"]
    PERFECT_FIFTH: ClassVar["Interval"]
    MINOR_SIXTH: ClassVar["Interval"]
    MAJOR_SIXTH: ClassVar["Interval"]
    MINOR_SEVENTH: ClassVar["Interval"]
    MAJOR_SEVENTH: ClassVar["Interval"]
    OCTAVE: ClassVar["Interval"]
    MINOR_NINTH: ClassVar["Interval"]
    MAJOR_NINTH: ClassVar["Interval"]
    MINOR_TENTH: ClassVar["Interval"]
    MAJOR_TENTH: ClassVar["Interval"]
    PERFECT_ELEVENTH: ClassVar["Interval"]
    AUGMENTED_ELEVENTH: ClassVar["Interval"]
    PERFECT_TWELFTH: ClassVar["Interval"]
    MINOR_THIRTEENTH: ClassVar["Interval"]
    MAJOR_THIRTEENTH: ClassVar["Interval"]

    def __repr__(self) -> str:
        return f"Interval({self.semitones}, {self.name!r})"

    @classmethod
    def from_semitones(cls, semitones: int) -> 'Interval':
        """Create interval from semitone count.

        Args:
            semitones: Number of semitones (0-24 typically)

        Returns:
            Matching Interval constant
        """
        semitones = semitones % 12  # Normalize to one octave
        for interval in INTERVALS:
            if interval.semitones == semitones:
                return interval
        # Default to chromatic interval
        return cls(semitones, IntervalQuality.AUGMENTED, semitones, f"{semitones} semitones")

    @classmethod
    def between(cls, note1: 'Note', note2: 'Note') -> 'Interval':
        """Calculate interval between two notes.

        Args:
            note1: First note
            note2: Second note

        Returns:
            Interval between the notes
        """
        semitones = abs(note2.midi - note1.midi)
        return cls.from_semitones(semitones)


# Standard intervals
Interval.UNISON = Interval(0, IntervalQuality.PERFECT, 1, "Unison")
Interval.MINOR_SECOND = Interval(1, IntervalQuality.MINOR, 2, "Minor Second")
Interval.MAJOR_SECOND = Interval(2, IntervalQuality.MAJOR, 2, "Major Second")
Interval.MINOR_THIRD = Interval(3, IntervalQuality.MINOR, 3, "Minor Third")
Interval.MAJOR_THIRD = Interval(4, IntervalQuality.MAJOR, 3, "Major Third")
Interval.PERFECT_FOURTH = Interval(5, IntervalQuality.PERFECT, 4, "Perfect Fourth")
Interval.TRITONE = Interval(6, IntervalQuality.AUGMENTED, 4, "Tritone")
Interval.PERFECT_FIFTH = Interval(7, IntervalQuality.PERFECT, 5, "Perfect Fifth")
Interval.MINOR_SIXTH = Interval(8, IntervalQuality.MINOR, 6, "Minor Sixth")
Interval.MAJOR_SIXTH = Interval(9, IntervalQuality.MAJOR, 6, "Major Sixth")
Interval.MINOR_SEVENTH = Interval(10, IntervalQuality.MINOR, 7, "Minor Seventh")
Interval.MAJOR_SEVENTH = Interval(11, IntervalQuality.MAJOR, 7, "Major Seventh")
Interval.OCTAVE = Interval(12, IntervalQuality.PERFECT, 8, "Octave")

# Compound intervals
Interval.MINOR_NINTH = Interval(13, IntervalQuality.MINOR, 9, "Minor Ninth")
Interval.MAJOR_NINTH = Interval(14, IntervalQuality.MAJOR, 9, "Major Ninth")
Interval.MINOR_TENTH = Interval(15, IntervalQuality.MINOR, 10, "Minor Tenth")
Interval.MAJOR_TENTH = Interval(16, IntervalQuality.MAJOR, 10, "Major Tenth")
Interval.PERFECT_ELEVENTH = Interval(17, IntervalQuality.PERFECT, 11, "Perfect Eleventh")
Interval.AUGMENTED_ELEVENTH = Interval(18, IntervalQuality.AUGMENTED, 11, "Augmented Eleventh")
Interval.PERFECT_TWELFTH = Interval(19, IntervalQuality.PERFECT, 12, "Perfect Twelfth")
Interval.MINOR_THIRTEENTH = Interval(20, IntervalQuality.MINOR, 13, "Minor Thirteenth")
Interval.MAJOR_THIRTEENTH = Interval(21, IntervalQuality.MAJOR, 13, "Major Thirteenth")

# List of standard intervals
INTERVALS = [
    Interval.UNISON, Interval.MINOR_SECOND, Interval.MAJOR_SECOND,
    Interval.MINOR_THIRD, Interval.MAJOR_THIRD, Interval.PERFECT_FOURTH,
    Interval.TRITONE, Interval.PERFECT_FIFTH, Interval.MINOR_SIXTH,
    Interval.MAJOR_SIXTH, Interval.MINOR_SEVENTH, Interval.MAJOR_SEVENTH,
    Interval.OCTAVE,
]


# ============================================================================
# Note Class
# ============================================================================


@dataclass
class Note:
    """Musical note with pitch class and octave.

    Attributes:
        name: Note name without octave (e.g., 'C', 'F#', 'Bb')
        octave: Octave number (-1 to 9 for MIDI range)
        velocity: Default velocity (0-127)

    Example:
        >>> note = Note('C', 4)  # Middle C
        >>> note.midi
        60
        >>> note.frequency
        261.63
        >>> note.transpose(7)  # G4
        Note('G', 4)
    """
    name: str
    octave: int = 4
    velocity: int = 100

    def __post_init__(self):
        # Normalize note name
        self.name = self.name.upper()
        if self.name in ENHARMONIC_MAP:
            self.name = ENHARMONIC_MAP[self.name]

        # Validate
        if self.name not in NOTE_NAMES:
            raise ValueError(f"Invalid note name: {self.name}")
        if not -1 <= self.octave <= 9:
            raise ValueError(f"Octave {self.octave} out of range (-1 to 9)")
        if not 0 <= self.velocity <= 127:
            raise ValueError(f"Velocity {self.velocity} out of range (0-127)")

    @property
    def midi(self) -> int:
        """MIDI note number (0-127)."""
        return note_name_to_midi(self.name, self.octave)

    @property
    def pitch_class(self) -> int:
        """Pitch class (0-11, where C=0)."""
        return NOTE_NAMES.index(self.name)

    @property
    def frequency(self) -> float:
        """Frequency in Hz (A4 = 440 Hz standard tuning)."""
        return float(440.0 * (2.0 ** ((self.midi - 69) / 12.0)))

    @classmethod
    def from_midi(cls, midi: int, velocity: int = 100, use_flats: bool = False) -> 'Note':
        """Create Note from MIDI number.

        Args:
            midi: MIDI note number (0-127)
            velocity: Note velocity (0-127)
            use_flats: Use flat names instead of sharps

        Returns:
            Note instance
        """
        if not 0 <= midi <= 127:
            raise ValueError(f"MIDI note {midi} out of range (0-127)")

        octave = (midi // 12) - 1
        pitch_class = midi % 12
        names = FLAT_NAMES if use_flats else NOTE_NAMES
        return cls(names[pitch_class], octave, velocity)

    def transpose(self, semitones: Union[int, Interval]) -> 'Note':
        """Transpose note by semitones or interval.

        Args:
            semitones: Number of semitones or Interval

        Returns:
            New transposed Note
        """
        if isinstance(semitones, Interval):
            semitones = semitones.semitones

        new_midi = self.midi + semitones
        if not 0 <= new_midi <= 127:
            raise ValueError(f"Transposed MIDI note {new_midi} out of range")

        return Note.from_midi(new_midi, self.velocity)

    def interval_to(self, other: 'Note') -> Interval:
        """Get interval to another note.

        Args:
            other: Target note

        Returns:
            Interval between this note and other
        """
        return Interval.between(self, other)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Note):
            return self.midi == other.midi
        return False

    def __hash__(self) -> int:
        return hash(self.midi)

    def __lt__(self, other: 'Note') -> bool:
        return self.midi < other.midi

    def __repr__(self) -> str:
        return f"Note({self.name!r}, {self.octave})"

    def __str__(self) -> str:
        return f"{self.name}{self.octave}"


# ============================================================================
# Scale Types and Class
# ============================================================================


class ScaleType(Enum):
    """Common scale types with interval patterns (semitones from root)."""
    # Diatonic scales
    MAJOR = (0, 2, 4, 5, 7, 9, 11)
    NATURAL_MINOR = (0, 2, 3, 5, 7, 8, 10)
    HARMONIC_MINOR = (0, 2, 3, 5, 7, 8, 11)
    MELODIC_MINOR = (0, 2, 3, 5, 7, 9, 11)

    # Modes
    IONIAN = (0, 2, 4, 5, 7, 9, 11)  # Same as major
    DORIAN = (0, 2, 3, 5, 7, 9, 10)
    PHRYGIAN = (0, 1, 3, 5, 7, 8, 10)
    LYDIAN = (0, 2, 4, 6, 7, 9, 11)
    MIXOLYDIAN = (0, 2, 4, 5, 7, 9, 10)
    AEOLIAN = (0, 2, 3, 5, 7, 8, 10)  # Same as natural minor
    LOCRIAN = (0, 1, 3, 5, 6, 8, 10)

    # Pentatonic scales
    MAJOR_PENTATONIC = (0, 2, 4, 7, 9)
    MINOR_PENTATONIC = (0, 3, 5, 7, 10)

    # Blues scales
    BLUES = (0, 3, 5, 6, 7, 10)
    MAJOR_BLUES = (0, 2, 3, 4, 7, 9)

    # Symmetric scales
    WHOLE_TONE = (0, 2, 4, 6, 8, 10)
    DIMINISHED = (0, 2, 3, 5, 6, 8, 9, 11)  # Half-whole
    DIMINISHED_WHOLE_HALF = (0, 1, 3, 4, 6, 7, 9, 10)  # Whole-half
    CHROMATIC = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    # Exotic scales
    HUNGARIAN_MINOR = (0, 2, 3, 6, 7, 8, 11)
    PHRYGIAN_DOMINANT = (0, 1, 4, 5, 7, 8, 10)  # Spanish/Jewish
    DOUBLE_HARMONIC = (0, 1, 4, 5, 7, 8, 11)  # Byzantine
    HIRAJOSHI = (0, 2, 3, 7, 8)  # Japanese
    IN_SEN = (0, 1, 5, 7, 10)  # Japanese
    IWATO = (0, 1, 5, 6, 10)  # Japanese
    ARABIAN = (0, 2, 4, 5, 6, 8, 10)
    PERSIAN = (0, 1, 4, 5, 6, 8, 11)
    BALINESE = (0, 1, 3, 7, 8)


class Mode(Enum):
    """Church modes (relative to major scale)."""
    IONIAN = 0      # I - Major
    DORIAN = 1      # II
    PHRYGIAN = 2    # III
    LYDIAN = 3      # IV
    MIXOLYDIAN = 4  # V
    AEOLIAN = 5     # VI - Natural minor
    LOCRIAN = 6     # VII


@dataclass
class Scale:
    """Musical scale from a root note.

    Attributes:
        root: Root note of the scale
        scale_type: Type of scale (intervals pattern)
        octaves: Number of octaves to generate

    Example:
        >>> scale = Scale(Note('C', 4), ScaleType.MAJOR)
        >>> notes = scale.get_notes()
        >>> [str(n) for n in notes]
        ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4']
        >>>
        >>> scale.contains(Note('E', 4))
        True
        >>> scale.degree(3)  # Third degree
        Note('E', 4)
    """
    root: Note
    scale_type: ScaleType
    octaves: int = 1

    @property
    def intervals(self) -> Tuple[int, ...]:
        """Semitone intervals from root."""
        result: Tuple[int, ...] = self.scale_type.value
        return result

    def get_notes(self, octaves: Optional[int] = None) -> List[Note]:
        """Get all notes in the scale.

        Args:
            octaves: Number of octaves (default: self.octaves)

        Returns:
            List of Notes in the scale
        """
        if octaves is None:
            octaves = self.octaves

        notes = []
        for octave_offset in range(octaves):
            for interval in self.intervals:
                midi = self.root.midi + interval + (octave_offset * 12)
                if 0 <= midi <= 127:
                    notes.append(Note.from_midi(midi, self.root.velocity))

        return notes

    def get_midi_notes(self, octaves: Optional[int] = None) -> List[int]:
        """Get MIDI note numbers in the scale.

        Args:
            octaves: Number of octaves

        Returns:
            List of MIDI note numbers
        """
        return [n.midi for n in self.get_notes(octaves)]

    def degree(self, degree: int) -> Note:
        """Get note at specific scale degree.

        Args:
            degree: Scale degree (1-based, 1 = root)

        Returns:
            Note at that degree
        """
        if degree < 1:
            raise ValueError(f"Degree must be >= 1, got {degree}")

        # Handle degrees beyond one octave
        octave_offset = (degree - 1) // len(self.intervals)
        degree_in_octave = (degree - 1) % len(self.intervals)

        interval = self.intervals[degree_in_octave]
        midi = self.root.midi + interval + (octave_offset * 12)

        if not 0 <= midi <= 127:
            raise ValueError(f"Degree {degree} results in MIDI {midi} out of range")

        return Note.from_midi(midi, self.root.velocity)

    def contains(self, note: Note) -> bool:
        """Check if note is in the scale.

        Args:
            note: Note to check

        Returns:
            True if note's pitch class is in the scale
        """
        root_pc = self.root.pitch_class
        note_pc = note.pitch_class
        relative_pc = (note_pc - root_pc) % 12
        return relative_pc in self.intervals

    def harmonize(self, degree: int, chord_type: Optional['ChordType'] = None) -> 'Chord':
        """Build chord on scale degree.

        Args:
            degree: Scale degree (1-based)
            chord_type: Chord type, or None for diatonic triad

        Returns:
            Chord built on that degree
        """
        root = self.degree(degree)

        if chord_type is not None:
            return Chord(root, chord_type)

        # Build diatonic triad
        third = self.degree(degree + 2)
        fifth = self.degree(degree + 4)

        third_interval = (third.midi - root.midi) % 12
        fifth_interval = (fifth.midi - root.midi) % 12

        # Determine chord type from intervals
        if third_interval == 4 and fifth_interval == 7:
            return Chord(root, ChordType.MAJOR)
        elif third_interval == 3 and fifth_interval == 7:
            return Chord(root, ChordType.MINOR)
        elif third_interval == 3 and fifth_interval == 6:
            return Chord(root, ChordType.DIMINISHED)
        elif third_interval == 4 and fifth_interval == 8:
            return Chord(root, ChordType.AUGMENTED)
        else:
            return Chord(root, ChordType.MAJOR)  # Default

    def parallel(self, scale_type: ScaleType) -> 'Scale':
        """Get parallel scale (same root, different type).

        Args:
            scale_type: New scale type

        Returns:
            Parallel scale
        """
        return Scale(self.root, scale_type, self.octaves)

    def relative_minor(self) -> 'Scale':
        """Get relative minor scale.

        Returns:
            Relative minor scale (6th degree becomes root)
        """
        minor_root = self.degree(6)
        return Scale(minor_root, ScaleType.NATURAL_MINOR, self.octaves)

    def relative_major(self) -> 'Scale':
        """Get relative major scale.

        Returns:
            Relative major scale (3rd degree becomes root)
        """
        major_root = self.degree(3)
        return Scale(major_root, ScaleType.MAJOR, self.octaves)

    def __iter__(self) -> Iterator[Note]:
        """Iterate over scale notes."""
        return iter(self.get_notes())

    def __len__(self) -> int:
        """Number of notes in one octave of the scale."""
        return len(self.intervals)

    def __repr__(self) -> str:
        return f"Scale({self.root}, {self.scale_type.name})"


# ============================================================================
# Chord Types and Class
# ============================================================================


class ChordType(Enum):
    """Common chord types with interval patterns (semitones from root)."""
    # Triads
    MAJOR = (0, 4, 7)
    MINOR = (0, 3, 7)
    DIMINISHED = (0, 3, 6)
    AUGMENTED = (0, 4, 8)
    SUS2 = (0, 2, 7)
    SUS4 = (0, 5, 7)

    # Seventh chords
    MAJOR_7 = (0, 4, 7, 11)
    MINOR_7 = (0, 3, 7, 10)
    DOMINANT_7 = (0, 4, 7, 10)
    DIMINISHED_7 = (0, 3, 6, 9)
    HALF_DIMINISHED_7 = (0, 3, 6, 10)  # Minor 7 flat 5
    MINOR_MAJOR_7 = (0, 3, 7, 11)
    AUGMENTED_7 = (0, 4, 8, 10)
    AUGMENTED_MAJOR_7 = (0, 4, 8, 11)

    # Sixth chords
    MAJOR_6 = (0, 4, 7, 9)
    MINOR_6 = (0, 3, 7, 9)

    # Add chords
    ADD9 = (0, 4, 7, 14)
    ADD11 = (0, 4, 7, 17)
    MINOR_ADD9 = (0, 3, 7, 14)

    # Extended chords
    MAJOR_9 = (0, 4, 7, 11, 14)
    MINOR_9 = (0, 3, 7, 10, 14)
    DOMINANT_9 = (0, 4, 7, 10, 14)
    MAJOR_11 = (0, 4, 7, 11, 14, 17)
    MINOR_11 = (0, 3, 7, 10, 14, 17)
    DOMINANT_11 = (0, 4, 7, 10, 14, 17)
    MAJOR_13 = (0, 4, 7, 11, 14, 17, 21)
    MINOR_13 = (0, 3, 7, 10, 14, 17, 21)
    DOMINANT_13 = (0, 4, 7, 10, 14, 17, 21)

    # Altered chords
    DOMINANT_7_FLAT_5 = (0, 4, 6, 10)
    DOMINANT_7_SHARP_5 = (0, 4, 8, 10)
    DOMINANT_7_FLAT_9 = (0, 4, 7, 10, 13)
    DOMINANT_7_SHARP_9 = (0, 4, 7, 10, 15)
    DOMINANT_7_FLAT_5_FLAT_9 = (0, 4, 6, 10, 13)
    DOMINANT_7_SHARP_5_SHARP_9 = (0, 4, 8, 10, 15)

    # Power chord
    POWER = (0, 7)
    POWER_OCTAVE = (0, 7, 12)


# Chord symbol mappings for parsing
CHORD_SYMBOL_MAP = {
    '': ChordType.MAJOR,
    'maj': ChordType.MAJOR,
    'M': ChordType.MAJOR,
    'm': ChordType.MINOR,
    'min': ChordType.MINOR,
    '-': ChordType.MINOR,
    'dim': ChordType.DIMINISHED,
    'o': ChordType.DIMINISHED,
    'aug': ChordType.AUGMENTED,
    '+': ChordType.AUGMENTED,
    'sus2': ChordType.SUS2,
    'sus4': ChordType.SUS4,
    'sus': ChordType.SUS4,
    '7': ChordType.DOMINANT_7,
    'maj7': ChordType.MAJOR_7,
    'M7': ChordType.MAJOR_7,
    'm7': ChordType.MINOR_7,
    'min7': ChordType.MINOR_7,
    '-7': ChordType.MINOR_7,
    'dim7': ChordType.DIMINISHED_7,
    'o7': ChordType.DIMINISHED_7,
    'm7b5': ChordType.HALF_DIMINISHED_7,
    'mM7': ChordType.MINOR_MAJOR_7,
    '6': ChordType.MAJOR_6,
    'm6': ChordType.MINOR_6,
    '9': ChordType.DOMINANT_9,
    'maj9': ChordType.MAJOR_9,
    'm9': ChordType.MINOR_9,
    '11': ChordType.DOMINANT_11,
    '13': ChordType.DOMINANT_13,
    'add9': ChordType.ADD9,
    '5': ChordType.POWER,
}


@dataclass
class Chord:
    """Musical chord from a root note.

    Attributes:
        root: Root note of the chord
        chord_type: Type of chord (interval pattern)

    Example:
        >>> chord = Chord(Note('C', 4), ChordType.MAJOR_7)
        >>> chord.get_midi_notes()
        [60, 64, 67, 71]
        >>>
        >>> chord.inversion(1)
        Chord with E as bass note
    """
    root: Note
    chord_type: ChordType

    @property
    def intervals(self) -> Tuple[int, ...]:
        """Semitone intervals from root."""
        result: Tuple[int, ...] = self.chord_type.value
        return result

    def get_notes(self) -> List[Note]:
        """Get all notes in the chord.

        Returns:
            List of Notes
        """
        notes = []
        for interval in self.intervals:
            midi = self.root.midi + interval
            if 0 <= midi <= 127:
                notes.append(Note.from_midi(midi, self.root.velocity))
        return notes

    def get_midi_notes(self) -> List[int]:
        """Get MIDI note numbers in the chord.

        Returns:
            List of MIDI note numbers
        """
        return [n.midi for n in self.get_notes()]

    def inversion(self, n: int) -> 'Chord':
        """Get chord inversion.

        Args:
            n: Inversion number (0=root, 1=first, 2=second, etc.)

        Returns:
            New Chord representing the inversion

        Note:
            Returns a chord with reordered intervals, not a truly new voicing.
            For more complex voicings, use get_voicing().
        """
        notes = self.get_notes()
        if n >= len(notes):
            n = n % len(notes)

        # Rotate notes and adjust octaves
        rotated = notes[n:] + [Note.from_midi(note.midi + 12) for note in notes[:n]]

        # Create new chord from lowest note
        return Chord(rotated[0], self.chord_type)

    def get_voicing(self, voicing: List[int]) -> List[Note]:
        """Get specific voicing of the chord.

        Args:
            voicing: List of interval indices (0=root, 1=third, etc.)
                     Can include negative (below root) or >len (octave up)

        Returns:
            List of Notes in the specified voicing

        Example:
            >>> chord = Chord(Note('C', 4), ChordType.MAJOR_7)
            >>> # Drop-2 voicing: root, 5th, 7th, 3rd (octave up)
            >>> chord.get_voicing([0, 2, 3, 5])  # 5 = 1 + 4 (3rd up octave)
        """
        notes = []
        num_intervals = len(self.intervals)

        for idx in voicing:
            octave_offset = idx // num_intervals
            interval_idx = idx % num_intervals
            interval = self.intervals[interval_idx]
            midi = self.root.midi + interval + (octave_offset * 12)

            if 0 <= midi <= 127:
                notes.append(Note.from_midi(midi, self.root.velocity))

        return notes

    def transpose(self, semitones: int) -> 'Chord':
        """Transpose chord by semitones.

        Args:
            semitones: Number of semitones

        Returns:
            Transposed chord
        """
        new_root = self.root.transpose(semitones)
        return Chord(new_root, self.chord_type)

    @classmethod
    def from_symbol(cls, symbol: str, octave: int = 4) -> 'Chord':
        """Parse chord from symbol string.

        Args:
            symbol: Chord symbol (e.g., 'Cmaj7', 'F#m', 'Bb7')
            octave: Root note octave

        Returns:
            Chord instance

        Example:
            >>> Chord.from_symbol('Cmaj7')
            Chord(Note('C', 4), ChordType.MAJOR_7)
            >>> Chord.from_symbol('F#m')
            Chord(Note('F#', 4), ChordType.MINOR)
        """
        # Parse root note (handles sharps/flats)
        match = re.match(r'^([A-Ga-g][#b]?)(.*?)$', symbol)
        if not match:
            raise ValueError(f"Invalid chord symbol: {symbol}")

        root_name = match.group(1).upper()
        quality = match.group(2)

        # Handle enharmonic
        if root_name in ENHARMONIC_MAP:
            root_name = ENHARMONIC_MAP[root_name]

        root = Note(root_name, octave)

        # Look up chord type
        chord_type = CHORD_SYMBOL_MAP.get(quality)
        if chord_type is None:
            raise ValueError(f"Unknown chord quality: {quality}")

        return cls(root, chord_type)

    @property
    def symbol(self) -> str:
        """Get chord symbol string."""
        # Reverse lookup in symbol map
        for sym, ct in CHORD_SYMBOL_MAP.items():
            if ct == self.chord_type and sym:  # Skip empty string
                return f"{self.root.name}{sym}"
        return f"{self.root.name}"

    def __iter__(self) -> Iterator[Note]:
        """Iterate over chord notes."""
        return iter(self.get_notes())

    def __len__(self) -> int:
        """Number of notes in the chord."""
        return len(self.intervals)

    def __repr__(self) -> str:
        return f"Chord({self.root}, {self.chord_type.name})"

    def __str__(self) -> str:
        return self.symbol


# ============================================================================
# Chord Progression
# ============================================================================


@dataclass
class ChordProgression:
    """A sequence of chords.

    Attributes:
        chords: List of chords
        key: Key signature (root note name)
        scale_type: Scale type for the key

    Example:
        >>> prog = ChordProgression.from_numerals('C', ['I', 'V', 'vi', 'IV'])
        >>> [str(c) for c in prog.chords]
        ['Cmaj', 'Gmaj', 'Am', 'Fmaj']
    """
    chords: List[Chord]
    key: Optional[str] = None
    scale_type: ScaleType = ScaleType.MAJOR

    @classmethod
    def from_numerals(
        cls,
        key: str,
        numerals: List[str],
        octave: int = 4,
        scale_type: ScaleType = ScaleType.MAJOR,
    ) -> 'ChordProgression':
        """Create progression from Roman numeral notation.

        Args:
            key: Root note name (e.g., 'C', 'F#')
            numerals: List of Roman numerals (e.g., ['I', 'IV', 'V', 'I'])
            octave: Base octave
            scale_type: Scale type

        Returns:
            ChordProgression instance

        Example:
            >>> ChordProgression.from_numerals('C', ['I', 'V', 'vi', 'IV'])
        """
        numeral_map = {
            'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7,
            'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7,
        }

        root = Note(key, octave)
        scale = Scale(root, scale_type)
        chords = []

        for numeral in numerals:
            # Check for additional quality modifiers
            base_numeral = numeral.rstrip('o+7')
            is_minor = base_numeral.islower()
            is_dim = 'o' in numeral
            is_aug = '+' in numeral
            has_7th = '7' in numeral

            degree = numeral_map.get(base_numeral.upper())
            if degree is None:
                raise ValueError(f"Invalid numeral: {numeral}")

            chord_root = scale.degree(degree)

            # Determine chord type
            if is_dim:
                chord_type = ChordType.DIMINISHED_7 if has_7th else ChordType.DIMINISHED
            elif is_aug:
                chord_type = ChordType.AUGMENTED
            elif is_minor:
                chord_type = ChordType.MINOR_7 if has_7th else ChordType.MINOR
            else:
                if has_7th:
                    # V7 is dominant, others are major 7
                    chord_type = ChordType.DOMINANT_7 if degree == 5 else ChordType.MAJOR_7
                else:
                    chord_type = ChordType.MAJOR

            chords.append(Chord(chord_root, chord_type))

        return cls(chords, key, scale_type)

    @classmethod
    def from_symbols(cls, symbols: List[str], octave: int = 4) -> 'ChordProgression':
        """Create progression from chord symbols.

        Args:
            symbols: List of chord symbols (e.g., ['Cmaj7', 'Dm7', 'G7'])
            octave: Base octave

        Returns:
            ChordProgression instance
        """
        chords = [Chord.from_symbol(sym, octave) for sym in symbols]
        return cls(chords)

    def transpose(self, semitones: int) -> 'ChordProgression':
        """Transpose entire progression.

        Args:
            semitones: Number of semitones

        Returns:
            Transposed progression
        """
        new_chords = [c.transpose(semitones) for c in self.chords]
        new_key = None
        if self.key:
            root = Note(self.key, 4)
            new_root = root.transpose(semitones)
            new_key = new_root.name
        return ChordProgression(new_chords, new_key, self.scale_type)

    def __iter__(self) -> Iterator[Chord]:
        """Iterate over chords."""
        return iter(self.chords)

    def __len__(self) -> int:
        """Number of chords."""
        return len(self.chords)

    def __repr__(self) -> str:
        symbols = [str(c) for c in self.chords]
        return f"ChordProgression({symbols})"
