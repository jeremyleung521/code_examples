#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT

DIHED=$(mktemp)

COMMAND="           parm $WEST_SIM_ROOT/common_files/diala_nowat_ff14_hmr.prmtop \n"
COMMAND="${COMMAND} trajin $WEST_STRUCT_DATA_REF \n"
COMMAND="${COMMAND} multidihedral phi psi out $DIHED range360\n"
COMMAND="${COMMAND} go"

echo -e "${COMMAND}" | $CPPTRAJ

cat $DIHED | tail -n +2 | awk '{print $2,$3}' > $WEST_PCOORD_RETURN

cp $WEST_SIM_ROOT/common_files/diala_nowat_ff14_hmr.prmtop $WEST_TRAJECTORY_RETURN
cp $WEST_STRUCT_DATA_REF $WEST_TRAJECTORY_RETURN

cp $WEST_SIM_ROOT/common_files/diala_nowat_ff14_hmr.prmtop $WEST_TRAJECTORY_RETURN
cp $WEST_STRUCT_DATA_REF $WEST_TRAJECTORY_RETURN/parent.ncrst

rm $DIHED

if [ -n "$SEG_DEBUG" ] ; then
  head -v $WEST_PCOORD_RETURN
fi
