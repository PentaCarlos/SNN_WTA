@echo off
setlocal enabledelayedexpansion
set Seed=pairSTDP_S

cd ..

FOR %%s IN (10 31 41 68) DO (
    set "c=!Seed!%%s"
    echo !c!
    python Train.py -s %%s -f !c! -d 40000 -e 1
)