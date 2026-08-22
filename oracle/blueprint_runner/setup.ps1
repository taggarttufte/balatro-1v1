# Reproduce the vendored analyzers used to build mp/oracle/ground_truth (pinned commits).
# vendor/ is gitignored (see .gitignore here) -- never commit the clones.
#
#   powershell -ExecutionPolicy Bypass -File mp/oracle/blueprint_runner/setup.ps1
#
# Requires: git, node >= 20, npm.  Then:
#   cd mp/oracle/blueprint_runner/vendor/Blueprint
#   $env:BLUEPRINT_COMMIT = (git rev-parse HEAD)
#   npx vite-node ../../run_blueprint.ts          -- --seed-file ../../seeds.txt --antes 8 --cards 50 --buy-vouchers --out ../../_raw
#   npx vite-node ../../run_blueprint_faithful.ts -- --seed-file ../../seeds.txt --antes 8 --cards 50 --out ../../_raw
#   npx vite-node ../../check_fixtures.ts
#   cd ../..
#   node run_thesoul.js --seed-file seeds.txt --antes 8 --cards 50 --out _raw
#   python ../build_ground_truth.py
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$vendor = Join-Path $here "vendor"
New-Item -ItemType Directory -Force $vendor | Out-Null

$repos = @(
    @{ name = "Blueprint";            url = "https://github.com/miaklwalker/Blueprint.git";            sha = "62898ed03cf3ea32719d64852fae797753d5665e" },
    @{ name = "TheSoul";              url = "https://github.com/SpectralPack/TheSoul.git";             sha = "780c1c21d9c5c283f2229548ee715b5fc2578342" },
    @{ name = "Immolate";             url = "https://github.com/MathIsFun0/Immolate.git";              sha = "26f41efcc313f045bc8bdbf49e5851c56ac40b31" },
    @{ name = "balatro-seed-finder";  url = "https://github.com/izanagi1995/balatro-seed-finder.git"; sha = "b3a112f8e6678f463e5bdce4298a04810bbc1594" }
)
foreach ($r in $repos) {
    $dst = Join-Path $vendor $r.name
    if (-not (Test-Path $dst)) {
        git clone --filter=blob:none $r.url $dst
    }
    git -C $dst fetch --depth 1 origin $r.sha
    git -C $dst checkout --quiet $r.sha
    Write-Host ("{0} @ {1}" -f $r.name, (git -C $dst rev-parse --short HEAD))
}
Push-Location (Join-Path $vendor "Blueprint")
npm ci --no-audit --no-fund --loglevel=error
Pop-Location
Write-Host "done. vendor/ is ready (gitignored)."
