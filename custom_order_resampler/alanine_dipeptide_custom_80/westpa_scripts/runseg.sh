#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
  set -x
  env | sort
fi

cd $WEST_SIM_ROOT
mkdir -pv $WEST_CURRENT_SEG_DATA_REF
cd $WEST_CURRENT_SEG_DATA_REF

ln -sv $WEST_SIM_ROOT/common_files/diala_nowat_ff14_hmr.prmtop .

if [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_CONTINUES" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md.in > md.in
  ln -sv $WEST_PARENT_DATA_REF/seg.ncrst ./parent.ncrst
elif [ "$WEST_CURRENT_SEG_INITPOINT_TYPE" = "SEG_INITPOINT_NEWTRAJ" ]; then
  sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/common_files/md.in > md.in
  ln -sv $WEST_PARENT_DATA_REF ./parent.ncrst
fi

$SANDER -O -i md.in   -p diala_nowat_ff14_hmr.prmtop  -c parent.ncrst \
          -r seg.ncrst -x seg.nc      -o seg.log    -inf seg.nfo


DIHED=$(mktemp)
DIHED2=$(mktemp)
ALIGN=$(mktemp)

COMMAND="           parm diala_nowat_ff14_hmr.prmtop \n"
COMMAND="${COMMAND} trajin $WEST_CURRENT_SEG_DATA_REF/parent.ncrst\n"
COMMAND="${COMMAND} trajin $WEST_CURRENT_SEG_DATA_REF/seg.nc\n"
COMMAND="${COMMAND} reference $WEST_SIM_ROOT/common_files/diala_nowat_eq2.pdb [reference] \n"
COMMAND="${COMMAND} rms ALIGN (!@H=) reference out $ALIGN \n"
COMMAND="${COMMAND} multidihedral phi psi out $DIHED range360\n"
COMMAND="${COMMAND} multidihedral phi psi out $DIHED2 \n"
COMMAND="${COMMAND} go"

echo -e "${COMMAND}" | $CPPTRAJ

cat $DIHED | tail -n +2 | awk '{print $2,$3}' > $WEST_PCOORD_RETURN
cat $DIHED2 | tail -n +2 | awk '{print $2}' > $WEST_PHI_RETURN
cat $DIHED2 | tail -n +2 | awk '{print $3}' > $WEST_PSI_RETURN

python $WEST_SIM_ROOT/common_files/get_coord.py
cp coord.npy $WEST_COORD_RETURN

cp diala_nowat_ff14_hmr.prmtop $WEST_TRAJECTORY_RETURN
cp seg.nc $WEST_TRAJECTORY_RETURN

cp diala_nowat_ff14_hmr.prmtop $WEST_RESTART_RETURN
cp seg.ncrst  $WEST_RESTART_RETURN/parent.ncrst

cp seg.log $WEST_LOG_RETURN

# Clean Up
rm $DIHED $DIHED2 $ALIGN coord.npy diala_nowat_ff14_hmr.prmtop

if [ -n "$SEG_DEBUG" ] ; then
  head -v $WEST_PCOORD_RETURN
fi


