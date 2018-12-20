#! /bin/bash
#
# Copyright (C) 2018 Free Software Foundation,
# Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#
# main()
#
if [ $# -ne 1 ] ; then
	echo Usage: "`basename $0` [ TYPE ]"
	echo "TYPE  service type. flask or quart"
	exit 1
fi

CORENUM=$(cat /proc/cpuinfo | grep "processor" | wc -l)
TYPE=$1

if [ $TYPE == "flask" ]; then
    gunicorn -t 9999 -w $CORENUM -b 0.0.0.0:5001 --certfile=cert/cert.pem --keyfile=cert/key.pem web_service:app
else
    hypercorn -w $CORENUM -b 0.0.0.0:5001 --certfile=cert/cert.pem --keyfile=cert/key.pem quart_service:app
fi

