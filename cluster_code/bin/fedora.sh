leading_zero_fill() {
	printf "%0$1d\\n" "$2"
}
# Source IC
source /gluster/data/next/software/IC_311024/config_ic.sh
echo "IC sourced"

NUMBER=$4
LEADING_NUMBER=$(leading_zero_fill 4 "$NUMBER")
INFILE=$1
OUTFILE=$2
RB_FACT=$3
RUNNUMBER=$5
TIMESTAMP=$6
LDC=$7


SOPHFILE="${INFILE}/run_${RUNNUMBER}_${LEADING_NUMBER}_ldc${LDC}*.h5"
REBINFILE="${OUTFILE}/run_${RUNNUMBER}_${LEADING_NUMBER}_ldc${LDC}.h5"

echo "Running rebinner.."
echo "================"
echo "INFILE: ${SOPHFILE}"
echo "OUTFILE: ${REBINFILE}"
echo "REBIN FACTOR: ${RB_FACT}"
echo "================"

if [ ! -f ${REBINFILE} ]; then
	echo "File not found, processing as normal..."
	python3 /gluster/home/jwaiton/scripts/bin/fedora.py $SOPHFILE $REBINFILE $RB_FACT
fi

echo "DONE!!!"
