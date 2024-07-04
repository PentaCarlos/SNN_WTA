@echo off
setlocal enabledelayedexpansion
set Temp=Temp_It_
set Rand=1

FOR /L %%s IN (1000,1000,40000) DO (
    set "c=!Temp!%%s"
    python Validate.py -s !Rand! -f !c! -t "True" -m %%s
)