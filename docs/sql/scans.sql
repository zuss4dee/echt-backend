-- Run in Supabase SQL Editor.
--
-- Scan history for /api/analyze. This file is the authoritative record of the
-- `scans` schema, which until now existed only in the Supabase dashboard.
--
-- WHY THIS EXISTS
-- ---------------
-- `scans` had no owner column. RLS was enabled on the table but NO policies were
-- defined, which in Postgres means "deny everything": the frontend reads with the
-- anon key (frontend/lib/supabase/browser.ts), so the Recent Scans sidebar in
-- app/analyze/page.tsx silently returned zero rows for every user, forever.
-- Writes were unaffected because the backend uses the service-role key
-- (SUPABASE_KEY in backend/.env.example), which bypasses RLS entirely — hence a
-- table full of rows that no client could read.
--
-- The trap: the obvious fix for an empty sidebar is to add a permissive select
-- policy, which would have exposed every customer's scans to every other
-- customer, because there was no column to scope on. The owner column and the
-- policy therefore have to land together, and do, below.
--
-- This script is additive and idempotent — safe to re-run.

-- ---------------------------------------------------------------------------
-- 1. Table shape
-- ---------------------------------------------------------------------------
-- The table already exists in every deployed environment, so this block is a
-- no-op there; it exists so a fresh project can be provisioned from this file.
-- Column list mirrors the insert at backend/main.py (see `supabase.table("scans")`).
create table if not exists public.scans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users (id) on delete cascade,
  filename text,
  doc_type text,
  verdict text,
  forgery_score numeric,
  trust_score numeric,
  red_flags jsonb,
  created_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- 2. Ownership column
-- ---------------------------------------------------------------------------
-- Deliberately NULLABLE. Pre-existing rows have no recoverable owner (nothing
-- was ever written that could identify one), and a NOT NULL column cannot be
-- added to a populated table anyway. Those rows simply become invisible under
-- the policy below rather than being deleted, so nothing is lost and this
-- migration cannot fail. New rows always carry a user_id: it comes from the
-- verified JWT subject returned by `verify_user` in backend/main.py.
alter table public.scans
  add column if not exists user_id uuid references auth.users (id) on delete cascade;

-- Matches the client's access pattern exactly: filter on user_id, order by
-- created_at desc, limit 5.
create index if not exists scans_user_id_created_at_idx
  on public.scans (user_id, created_at desc);

-- ---------------------------------------------------------------------------
-- 3. Row level security
-- ---------------------------------------------------------------------------
-- Already enabled; restated so this file fully describes the table's state.
alter table public.scans enable row level security;

-- Read-your-own, mirroring the pattern in docs/sql/profiles.sql.
-- Dropped first so the script stays re-runnable.
drop policy if exists "Users can read own scans" on public.scans;
create policy "Users can read own scans"
  on public.scans for select
  to authenticated
  using (auth.uid() = user_id);

-- NOTE: there are deliberately NO insert/update/delete policies.
-- All writes go through the backend with the service role, which bypasses RLS
-- (same arrangement as the webhook writes in docs/sql/whop_entitlements.sql).
-- Omitting these policies means no browser client can ever create, alter or
-- destroy a scan record, including its own.

grant usage on schema public to anon, authenticated;
grant select on public.scans to authenticated;

comment on table public.scans is
  'Per-user scan history for /api/analyze. Written by the backend with the service role; read by the owning user only, via RLS.';
comment on column public.scans.user_id is
  'Owner. Supabase auth.users id, taken from the verified JWT subject at scan time. NULL only on rows predating this column.';

-- ---------------------------------------------------------------------------
-- 4. Follow-up — DO NOT RUN YET
-- ---------------------------------------------------------------------------
-- Once you are satisfied the unattributed legacy rows are disposable, purge them
-- and let the schema itself guarantee ownership from then on. Destructive: these
-- rows cannot be reconstructed.
--
--   delete from public.scans where user_id is null;
--   alter table public.scans alter column user_id set not null;
