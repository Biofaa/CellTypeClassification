#!/bin/bash

message='--Resources--'

# Loop over the list of nodes
for node in daneel01 daneel02 daneel03; do
    # Extract the available and assigned resources
    available_ncpus=$(pbsnodes $node | awk '/resources_available.ncpus/ {print $3}')
    available_ngpus=$(pbsnodes $node | awk '/resources_available.ngpus/ {print $3}')
    assigned_ncpus=$(pbsnodes $node | awk '/resources_assigned.ncpus/ {print $3}')
    assigned_ngpus=$(pbsnodes $node | awk '/resources_assigned.ngpus/ {print $3}')