import os


def sound_alert():
    """
    Plays a sound when called
    :return:
    """
    duration = 1  # second
    freq = 440  # Hz
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))