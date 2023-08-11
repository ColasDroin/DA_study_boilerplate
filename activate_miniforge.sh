# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if  the job is launched from htcondor
if [[ $PWD == *"pool/condor"* ]]
# source the make_miniforge.sh script, concatenating the path to the script directory
then
    source $SCRIPT_DIR/make_miniforge.sh
# Only activate environment 
else
    source $SCRIPT_DIR/miniforge/bin/activate
fi