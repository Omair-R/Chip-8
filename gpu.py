from pygame import Color
import typing

delaytime: int = 1
scale: int = 15
width: int = 64 * scale
height: int = 32 * scale


class Colors:
    black = Color(0, 0, 0)
    white = Color(255, 255, 255)


def scale_rect(pos: typing.List[int]) -> typing.List[int]:
    return [i * scale for i in pos] + [scale, scale]
