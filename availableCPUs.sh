#!/bin/bash

message='--Resources--'

# Loop over the list of nodes
for node in hal01 hal02 daneel01 daneel02 daneel03; do
    # Extract the available and assigned resources
    available_ncpus=$(pbsnodes $node | awk '/resources_available.ncpus/ {print $3}')
    available_ngpus=$(pbsnodes $node | awk '/resources_available.ngpus/ {print $3}')
    assigned_ncpus=$(pbsnodes $node | awk '/resources_assigned.ncpus/ {print $3}')
    assigned_ngpus=$(pbsnodes $node | awk '/resources_assigned.ngpus/ {print $3}')

    # Compute the difference between available and assigned resources
    diff_ncpus=$((available_ncpus - assigned_ncpus))
    diff_ngpus=$((available_ngpus - assigned_ngpus))

    # Print the results
    message="$message"$'\n\n'" Node:$node - Av CPUs:$diff_ncpus - Av GPUs:$diff_ngpus"
done

# Escape special Markdown characters
message="${message//_/\\_}"

echo "$message"