from types import SimpleNamespace

default_colour = "#089682"

transparency = True

dpi = 600

# global colours
colour_dict = {
    "ax": "#555555",
    "labels": "#424242",
    "default": "#089682",
    "map": "RdYlBu",
    "highlight": "#FEB308",
    "rainbow": ["#FF6663", "#FEB144", "#FDFD97", "#9EE09E", "#9EC1CF", "#CC99C9"],
    "ln": "rgb(116, 62, 122)",
    "f1": "rgb(251, 218, 230)",
    "f2": "rgb(255, 241, 194)",
    "A769": ["#e76f51", "#A23216"],
    "PF739": ["#f4a261", "#994B0B"],
    "SC4": ["#e9c46a", "#A07918"],
    "MT47": ["#2a9d8f", "#1E7167"],
    "MK87": ["#264653", "#13242A"],
}

nspc = SimpleNamespace(**colour_dict)
