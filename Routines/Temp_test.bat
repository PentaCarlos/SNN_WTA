@echo off
setlocal enabledelayedexpansion
set Temp=Temp_It_
set Rand=0

cd ..

FOR /L %%s IN (1000,1000,5000) DO (
    set "c=!Temp!%%s"
    python Validate.py -s !Rand! -f !c! -t "True" -m %%s -gb "True" -n "True"
)

FOR /L %%s IN (10000,5000,40000) DO (
    set "c=!Temp!%%s"
    python Validate.py -s !Rand! -f !c! -t "True" -m %%s -gb "True" -n "True"
)