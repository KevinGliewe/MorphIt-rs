<#
.SYNOPSIS
  Run this repository's GitHub Actions locally with act (https://github.com/nektos/act).

.DESCRIPTION
  Linux jobs run in Docker (Docker Desktop must be running); Windows and macOS
  jobs cannot run locally, use -DryRun to check the job graph. Every event file
  under tools/ci/events carries "act": true, which the workflows use to skip
  deployments, the GitHub Release and the real crates.io upload.

  Behind a TLS-intercepting proxy, point MORPHIT_CA_PEM at the proxy's CA chain
  (a PEM kept outside the repository); it is mounted into the job containers
  and trusted by the setup action.

.EXAMPLE
  tools/act.ps1 ci -Job fmt
  tools/act.ps1 ci -Job test -Matrix os:ubuntu-latest
  tools/act.ps1 pages -Job build
  tools/act.ps1 release        # Linux x64 leg, web and crates dry run; other platforms are skipped
  tools/act.ps1 release -Event tag -DryRun
#>
param(
    [Parameter(Mandatory)][ValidateSet('ci', 'pages', 'release')][string]$Workflow,
    [string]$Job,
    # Event file in tools/ci/events (push, pull_request, tag, dispatch, release_dispatch).
    [string]$Event,
    # Matrix filters such as os:ubuntu-latest (act --matrix).
    [string[]]$Matrix,
    [switch]$DryRun,
    # Further act arguments.
    [Parameter(ValueFromRemainingArguments)][string[]]$Rest
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Event) {
    $Event = @{ ci = 'push'; pages = 'dispatch'; release = 'release_dispatch' }[$Workflow]
}
$eventName = switch ($Event) {
    'pull_request' { 'pull_request' }
    { $_ -in 'push', 'tag' } { 'push' }
    default { 'workflow_dispatch' }
}

$image = 'catthehacker/ubuntu:act-22.04'
$act = @(
    $eventName,
    '-W', ".github/workflows/$Workflow.yml",
    '-e', "tools/ci/events/$Event.json",
    '-P', "ubuntu-latest=$image",
    '-P', "ubuntu-22.04=$image"
)
# Pull the runner image once instead of for every job (Docker Hub rate limits).
if (docker image ls -q $image) { $act += '--pull=false' }
if ($Job) { $act += @('-j', $Job) }
foreach ($m in $Matrix) { $act += @('--matrix', $m) }
if ($DryRun) { $act += '-n' } else { $act += @('--artifact-server-path', "$root/target/act-artifacts") }
if ($env:MORPHIT_CA_PEM) {
    $pem = (Resolve-Path $env:MORPHIT_CA_PEM).Path
    $act += @('--container-options', "-v `"${pem}:/tmp/extra-ca.crt:ro`"")
}
if ($Rest) { $act += $Rest }

Write-Host "act $($act -join ' ')"
& act @act
exit $LASTEXITCODE
