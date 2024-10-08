Disable ImageSegmentation because it require Pymatting and that package take ~30s to import at the first time.
Ref https://github.com/pymatting/pymatting/issues/78.
We can setup a github action runner with the same architecture as the server to pre-import the package, but it's not worth the effort for now.