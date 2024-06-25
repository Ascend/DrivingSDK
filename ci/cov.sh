#!/bin/bash
CUR_DIR=$(dirname $(readlink -f $0))
NEW_BRANCH_PATH=$(readlink -f ${CUR_DIR}/..)
MASTER_BRANCH_PATH=$(readlink -f ${NEW_BRANCH_PATH}/../mx_driving)

#生成覆盖率
diff -r -N -x ".git" -x "*.doc" -x "*.json" -x "*.h" -x "*.py" -x "*.so" -x "*.info" -x "*.o" -u ${MASTER_BRANCH_PATH} ${NEW_BRANCH_PATH} >> diff.txt
lcov --rc lcov_branch_coverage=1 -c -d ${CUR_DIR}/.. -o coverage.info
STRIP=$(echo ${NEW_BRANCH_PATH}/ | tr -cd '/' | wc -c)
addlcov --diff coverage.info diff.txt -o inc.info --strip $STRIP --path ${NEW_BRANCH_PATH}
if [ -e inc.info ]
then
    genhtml inc.info -o output
else
    echo "inc.info not exist"
fi
