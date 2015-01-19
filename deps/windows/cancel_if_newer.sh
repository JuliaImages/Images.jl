#!/bin/sh

# Fail fast on AppVeyor if there are newer pending commits in this PR
curlflags="curl --retry 10 -k -L -y 5"
if [ -n "$APPVEYOR_PULL_REQUEST_NUMBER" ]; then
  # download a handy cli json parser
  if ! [ -e jq.exe ]; then
    $curlflags -O http://stedolan.github.io/jq/download/win64/jq.exe
  fi
  av_api_url="https://ci.appveyor.com/api/projects/timholy/images-jl/history?recordsNumber=50"
  query=".builds | map(select(.pullRequestId == \"$APPVEYOR_PULL_REQUEST_NUMBER\"))[0].buildNumber"
  latestbuild="$(curl $av_api_url | ./jq "$query")"
  if [ -n "$latestbuild" -a "$latestbuild" != "null" -a "$latestbuild" != "$APPVEYOR_BUILD_NUMBER" ]; then
    echo "There are newer queued builds for this pull request, failing early."
    exit 1
  fi
fi
