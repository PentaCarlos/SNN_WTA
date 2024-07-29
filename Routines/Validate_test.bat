@echo off
setlocal enabledelayedexpansion
set Seed=pairSTDP_S

cd ..

FOR %%s IN (10 31 41 68) DO (
    set "c=!Seed!%%s"
    python Validate.py -s %%s -f !c! -m 40000
)