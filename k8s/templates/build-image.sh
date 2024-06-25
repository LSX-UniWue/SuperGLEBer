#!/usr/bin/env bash
# has to be executed at the root directory of the project, like so:

# {{lastname}}@vingilot /a/c/p/p/Superkleber> pwd
# /autofs/ceph-{{affiliation}}/{{lastname}}/projects/Superkleber
# {{lastname}}@vingilot /a/c/p/p/Superkleber> bash k8s/{{lastname}}/build-image.sh

buildah bud -t "{{image_name}}" -f k8s/{{lastname}}/Dockerfile
buildah push "{{image_name}}"
