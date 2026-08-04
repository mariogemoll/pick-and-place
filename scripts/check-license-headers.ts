#!/usr/bin/env node
// SPDX-FileCopyrightText: 2026 Mario Gemoll
// SPDX-License-Identifier: 0BSD

import { execFileSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

import { config } from './license-config.js';

interface ValidationError {
	file: string;
	expected: string;
	actual: string | null;
	field?: 'copyright' | 'license';
}

function getCommentStyle(filePath: string): { start: string; end: string } | null {
  const ext = path.extname(filePath);

  switch (ext) {
  case '.ts':
  case '.tsx':
  case '.js':
  case '.jsx':
  case '.mjs':
    return { start: '// ', end: '' };
  case '.py':
  case '.sh':
    return { start: '# ', end: '' };
  case '.css':
    return { start: '/* ', end: ' */' };
  case '.html':
  case '.md':
    return { start: '<!-- ', end: ' -->' };
  default:
    return null;
  }
}

function matchesPattern(filePath: string, pattern: string): boolean {
  // Simple glob matching - supports **/*.ext and specific paths
  const normalizedPath = filePath.replace(/\\/g, '/');
  const normalizedPattern = pattern.replace(/\\/g, '/');

  // Exact match
  if (normalizedPath === normalizedPattern) {
    return true;
  }

  // Pattern like **/*.ext
  if (normalizedPattern.includes('*')) {
    const regexPattern = normalizedPattern
      .replace(/\*\*\//g, '__GLOBSTAR_SLASH__')  // **/ -> placeholder
      .replace(/\/\*\*/g, '__SLASH_GLOBSTAR__')  // /** -> placeholder
      .replace(/\*\*/g, '__GLOBSTAR__')          // ** -> placeholder
      .replace(/\*/g, '__STAR__')                // * -> placeholder
      .replace(/\./g, '\\.')                     // Escape dots
      .replace(/__GLOBSTAR_SLASH__/g, '(.*/)?')  // **/ matches zero or more directory segments
      .replace(/__SLASH_GLOBSTAR__/g, '(/.*)?')  // /** matches dir segments at end
      .replace(/__GLOBSTAR__/g, '.*')            // ** in other positions matches any characters
      .replace(/__STAR__/g, '[^/]*');            // * matches any characters except /
    const regex = new RegExp(`^${regexPattern}$`);
    return regex.test(normalizedPath);
  }

  return false;
}

function getLicenseForFile(filePath: string): string | null {
  const relativePath = path.relative(process.cwd(), filePath);

  // Check each rule in order
  for (const rule of config.rules) {
    for (const pattern of rule.files) {
      if (matchesPattern(relativePath, pattern)) {
        return rule.license;
      }
    }
  }

  return null;
}

function extractLicenseFromContent(content: string): string | null {
  // Check first 5 lines for SPDX identifier
  const lines = content.split('\n').slice(0, 5);

  for (const line of lines) {
    const match = /SPDX-License-Identifier:\s*([^\s-]+(?:-[^\s-]+)*)/.exec(line);
    if (match) {
      return match[1];
    }
  }

  return null;
}

function extractCopyrightFromContent(content: string): string | null {
  // Check first 5 lines for SPDX copyright
  const lines = content.split('\n').slice(0, 5);

  for (const line of lines) {
    const match = /SPDX-FileCopyrightText:\s*(.+?)(?:\s*(?:-->|\*\/|$))/.exec(line);
    if (match) {
      return match[1].trim();
    }
  }

  return null;
}

function hasEmptyLineAfterHeaders(content: string): boolean {
  // Check if there's an empty line after the SPDX headers
  const lines = content.split('\n');
  let licenseLineIndex = -1;

  // Find the line with SPDX-License-Identifier (should come after FileCopyrightText)
  for (let i = 0; i < Math.min(lines.length, 10); i++) {
    if (lines[i].includes('SPDX-License-Identifier:')) {
      licenseLineIndex = i;
      break;
    }
  }

  if (licenseLineIndex === -1) {
    return false;
  }

  // Check if the next line is empty
  const nextLineIndex = licenseLineIndex + 1;
  if (nextLineIndex < lines.length) {
    return lines[nextLineIndex].trim() === '';
  }

  return false;
}

function validateNotebook(filePath: string, expectedLicense: string): ValidationError | null {
  const content = fs.readFileSync(filePath, 'utf-8');
  const relativePath = path.relative(process.cwd(), filePath);
  const validCopyrights = ['2026 Mario Gemoll'];

  let notebook: { cells?: { cell_type: string; source: string[] }[] };
  try {
    notebook = JSON.parse(content);
  } catch {
    return { file: relativePath, expected: 'valid JSON notebook', actual: 'parse error', field: 'license' };
  }

  const firstCell = notebook.cells?.[0];
  if (!firstCell) {
    return { file: relativePath, expected: validCopyrights.join(' or '), actual: null, field: 'copyright' };
  }

  const cellSource = firstCell.source.join('');
  const commentStyle = firstCell.cell_type === 'markdown'
    ? { start: '<!-- ', end: ' -->' }
    : { start: '# ', end: '' };

  // Reuse the text-based extractors on the cell source
  // but we need to check comment style matches what's expected
  const actualCopyright = extractCopyrightFromContent(cellSource);
  const actualLicense = extractLicenseFromContent(cellSource);

  // Verify the comment prefix matches the cell type
  const firstLine = cellSource.split('\n')[0];
  if (!firstLine.startsWith(commentStyle.start)) {
    return {
      file: relativePath,
      expected: `${commentStyle.start}SPDX-FileCopyrightText: ...`,
      actual: firstLine || null,
      field: 'copyright'
    };
  }

  if (actualCopyright === null) {
    return { file: relativePath, expected: validCopyrights.join(' or '), actual: null, field: 'copyright' };
  }
  if (!validCopyrights.includes(actualCopyright)) {
    return { file: relativePath, expected: validCopyrights.join(' or '), actual: actualCopyright, field: 'copyright' };
  }
  if (actualLicense === null) {
    return { file: relativePath, expected: expectedLicense, actual: null, field: 'license' };
  }
  if (actualLicense !== expectedLicense) {
    return { file: relativePath, expected: expectedLicense, actual: actualLicense, field: 'license' };
  }

  return null;
}

function validateFile(filePath: string, expectedLicense: string): ValidationError | null {
  const content = fs.readFileSync(filePath, 'utf-8');
  const actualLicense = extractLicenseFromContent(content);
  const actualCopyright = extractCopyrightFromContent(content);
  const relativePath = path.relative(process.cwd(), filePath);

  const validCopyrights = ['2026 Mario Gemoll'];

  // Check copyright
  if (actualCopyright === null) {
    return {
      file: relativePath,
      expected: validCopyrights.join(' or '),
      actual: null,
      field: 'copyright'
    };
  }

  if (!validCopyrights.includes(actualCopyright)) {
    return {
      file: relativePath,
      expected: validCopyrights.join(' or '),
      actual: actualCopyright,
      field: 'copyright'
    };
  }

  // Check license
  if (actualLicense === null) {
    return {
      file: relativePath,
      expected: expectedLicense,
      actual: null,
      field: 'license'
    };
  }

  if (actualLicense !== expectedLicense) {
    return {
      file: relativePath,
      expected: expectedLicense,
      actual: actualLicense,
      field: 'license'
    };
  }

  // Check for empty line after headers
  if (!hasEmptyLineAfterHeaders(content)) {
    return {
      file: relativePath,
      expected: 'empty line after SPDX headers',
      actual: 'no empty line found',
      field: 'license'
    };
  }

  return null;
}

function checkFile(fullPath: string, errors: ValidationError[]): void {
  if (config.excludedFiles.includes(path.basename(fullPath))) {
    return;
  }

  const expectedLicense = getLicenseForFile(fullPath);
  if (expectedLicense === null) {
    return;
  }

  if (path.extname(fullPath) === '.ipynb') {
    const error = validateNotebook(fullPath, expectedLicense);
    if (error) errors.push(error);
    return;
  }

  if (getCommentStyle(fullPath) !== null) {
    const error = validateFile(fullPath, expectedLicense);
    if (error) errors.push(error);
  }
}

/**
 * Enumerate every file git would consider part of the project: tracked files
 * plus new ones not yet added, minus anything ignored. Walking the filesystem
 * instead would descend into local datasets, training output, and working
 * notes, which are ignored precisely because they are not ours to police —
 * while `--others` keeps a brand-new file from escaping the check simply
 * because it has not been committed yet.
 */
function projectFiles(rootDir: string): string[] {
  const stdout = execFileSync(
    'git',
    ['ls-files', '-z', '--cached', '--others', '--exclude-standard'],
    { cwd: rootDir, maxBuffer: 64 * 1024 * 1024 }
  ).toString('utf8');

  return stdout
    .split('\0')
    .filter((entry) => entry.length > 0)
    .filter((entry) => !entry.split('/').some((part) => config.excludedDirs.includes(part)));
}

function main(): void {
  const rootDir = process.cwd();
  const errors: ValidationError[] = [];

  console.log('Checking SPDX license headers...\n');
  for (const relativePath of projectFiles(rootDir)) {
    const fullPath = path.join(rootDir, relativePath);
    if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
      checkFile(fullPath, errors);
    }
  }

  if (errors.length === 0) {
    console.log('✓ All files have correct license headers!');
    process.exit(0);
  } else {
    console.error(`✗ Found ${errors.length} file(s) with incorrect or missing license headers:\n`);

    for (const error of errors) {
      const fieldName = error.field === 'copyright' ? 'copyright' : 'license';
      if (error.actual === null) {
        console.error(`  ${error.file}`);
        console.error(`    Missing ${fieldName} header (expected: ${error.expected})`);
      } else {
        console.error(`  ${error.file}`);
        console.error(`    Expected ${fieldName}: ${error.expected}`);
        console.error(`    Found ${fieldName}:    ${error.actual}`);
      }
      console.error('');
    }

    console.error('Fix the headers listed above and re-run this check.');
    process.exit(1);
  }
}

main();
