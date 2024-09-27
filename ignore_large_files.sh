#!/bin/bash

# Find and list large files
LARGE_FILES=$(find . -type f -size +49M)

# Check if there are any large files
if [ -z "$LARGE_FILES" ]; then
    echo "No large files found."
    exit 0
fi

# Create or edit .gitignore
echo "Adding large files to .gitignore..."
for FILE in $LARGE_FILES; do
    echo "$FILE" >> .gitignore
    git rm --cached "$FILE"
done

# Commit changes
git add .gitignore

echo "The following large files were ignored:"
echo "$LARGE_FILES"
