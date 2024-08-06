Guide for setting up simulation for Isaac Sim:

Install isaac sim on computer - minimum version 2023 with Cortex World 
compatibility

in home/user/.local/share/ov/pkg/isaac_sim-2023.1 we have root directory of 
isaac sim

open terminal in this directory a few packages has to be installed

./python.sh -m pip install 'skrl'
./python.sh -m pip install 'pygame'
./python.sh -m pip install 'opencv-python'
./python.sh -m pip install 'gymnasium'
./python.sh -m pip install 'tqdm'
(more might be needed but it should say what it needs)

create a modules directory and insert contents from modules directory in 
simulation_setup

Now move into /exts/omni.isaac.cortex.sample_behaviors/omni/isaac/cortex/sample_behaviors/ur10/

and move all contents from the ur10 directory in simulation setup

Go back to root directory and go into /standalone_examples/api/omni.isaac.cortex
And move contents from omni.isaac.cortex in simulation_setup

Go back to root and create agent directory
Insert content from agent directory in simulation_setup

Go back to root create usd_files directory
Insert contents from usd_files in simulation_setup

Now to run everything:

go back to root and open terminal

run:
./python.sh standalone_examples/api/omni.isaac.cortex/bpp_world.py

when isaac sim is loaded completely press the play button
