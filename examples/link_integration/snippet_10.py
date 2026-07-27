#!/usr/bin/env python3
"""Multiple Players Synchronized."""

# --8<-- [start:example]
from coremusic import link
from coremusic.base import AudioPlayer

# Share one Link session across multiple players
with link.LinkSession(bpm=120.0) as session:
    # Create multiple players
    player1 = AudioPlayer(link_session=session)
    player2 = AudioPlayer(link_session=session)

    player1.load_file("drums.wav")
    player2.load_file("input.wav")

    player1.setup_output()
    player2.setup_output()

    # Both players see same Link timing
    timing1 = player1.get_link_timing()
    timing2 = player2.get_link_timing()

    assert timing1['tempo'] == timing2['tempo']
    assert abs(timing1['beat'] - timing2['beat']) < 0.01

    # Start both (synchronized via Link)
    player1.play()
    player2.play()
    player1.start()
    player2.start()
# --8<-- [end:example]
