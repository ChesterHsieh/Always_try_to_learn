"""HL controllers.

Version-immutability policy (capability hl-lunar-lander spec):
Each baseline_v{n}.py is a first-class, immutable artefact of the HL iteration
trajectory. Once merged to main, a controller's behaviour MUST NOT be edited in
place — a new iteration creates baseline_v{n+1}.py instead. Only behaviour-neutral
edits (docstring, typo, type hint) are allowed on an existing version.
"""
