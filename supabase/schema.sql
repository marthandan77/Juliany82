-- Run once in the Supabase SQL Editor for the Juliany82 forecast ledger.
create table if not exists public.forecasts (
    forecast_id text primary key,
    record jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists public.outcomes (
    forecast_id text primary key references public.forecasts (forecast_id),
    record jsonb not null,
    created_at timestamptz not null default now()
);

alter table public.forecasts enable row level security;
alter table public.outcomes enable row level security;

-- The app uses the server-side service_role key, which bypasses RLS.
-- Never put that key in browser code, source control, or a public repository.
grant all on public.forecasts to service_role;
grant all on public.outcomes to service_role;
