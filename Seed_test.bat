@echo off
setlocal enabledelayedexpansion
set Seed=pairSTDP_S

FOR /L %%s IN (0,1,4) DO (
    set "c=!Temp!%%s"
    python Validate.py -s %%s -f !c! -m 40000
)