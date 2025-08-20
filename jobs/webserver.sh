#!/bin/bash
#$ -pe smp 1             # We only need 1 core for a web server
#$ -m be
#$ -l h_vmem=1G          # 1G of memory is more than enough
#$ -l h_rt=1:0:0         # Let's run for 1 hour (adjust if needed)
#$ -cwd
#$ -o web_server.o$JOB_ID # Use a unique output file name
#$ -e web_server.e$JOB_ID # Use a unique error file name

# --- CONFIGURATION TO EDIT ---

# 1. Set the root directory to where your plots and results are.
#    Since you run the script from the 'AML' directory and plots are saved there,
#    we can use the current working directory.
WEB_ROOT_DIR=$(pwd) 

# 2. Choose ports. 8000 and 9000 are fine unless they are in use.
SERVER_PORT=8000
LOCAL_FORWARD_PORT=9000

# 3. Set your HPC login details.
HPC_SSH_ALIAS="apocrita" # Or whatever you use in your ~/.ssh/config
YOUR_HPC_USERNAME="ec24713" #<-- CHANGE HERE: Use your username.
HPC_LOGIN_NODE_ADDRESS="login.hpc.qmul.ac.uk" # This is only correct if you want to connect to Apocrita


echo "Job submitted: $JOB_ID"
echo "Running on host: $(hostname)"
echo "Loading environment..."


if [ ! -d "$WEB_ROOT_DIR" ]; then
    echo "Error: Web root directory '$WEB_ROOT_DIR' not found!"
    exit 1
fi

NODE_HOSTNAME=$(hostname)

echo "--------------------------------------------------------------------"
echo "Python HTTP Server is starting up..."
echo "Serving files from: $WEB_ROOT_DIR"
echo "Running on HPC compute node: $NODE_HOSTNAME:$SERVER_PORT"
echo ""
echo ">>> ACTION REQUIRED ON YOUR LOCAL LAPTOP <<<"
echo "Open a NEW terminal on your own computer (not on the HPC) and run this exact command:"
echo ""
echo "   ssh -N -L ${LOCAL_FORWARD_PORT}:${NODE_HOSTNAME}:${SERVER_PORT} ${YOUR_HPC_USERNAME}@${HPC_LOGIN_NODE_ADDRESS}"
echo ""
echo "After running the command above, open this URL in your web browser:"
echo ""
echo "   http://localhost:${LOCAL_FORWARD_PORT}"
echo ""
echo "This job will keep the server running. To stop it, run: qdel $JOB_ID"
echo "--------------------------------------------------------------------"

cd "$WEB_ROOT_DIR" || exit 1

# Start the Python 3 HTTP server.
python3 -m http.server "$SERVER_PORT"

echo "Web server job $JOB_ID finished."
