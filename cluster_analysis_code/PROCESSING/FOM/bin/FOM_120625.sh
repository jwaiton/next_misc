leading_zero_fill() {
	printf "%0$1d\\n" "$2"
}
# Source IC
# conda activate the FOM_env here dummy!
echo "IC sourced"

RUNNUMBER=15153
TIMESTAMP=210625
CITY=thekla
LOWER_E=1.5108
UPPER_E=1.8108
LOWER_Z=20
UPPER_Z=1170
R_LIM=450


echo "Running FOM calculator 040625.."
echo "================"
echo "RUN NUMBER    : ${RUNNUMBER}"
echo "TIMESTAMP     : ${TIMESTAMP}"
echo "LOWER, UPPER E: ${LOWER_E}, ${UPPER_E}"
echo "LOWER, UPPER Z: ${LOWER_Z}, ${UPPER_Z}"
echo "R_LIM         : ${R_LIM}"
echo "================"

python3 FOM_120625.py $RUNNUMBER $TIMESTAMP $CITY $LOWER_E $UPPER_E $LOWER_Z $UPPER_Z $R_LIM > $RUNNUMBER > "${RUNNUMBER}_${TIMESTAMP}"


echo "DONE!!!"
