-- Optional: run after profiles.sql to store Polar subscription linkage on each agency row.

alter table public.profiles
  add column if not exists polar_checkout_id text,
  add column if not exists polar_customer_id text,
  add column if not exists subscription_status text default 'active';

create index if not exists profiles_polar_checkout_id_idx
  on public.profiles (polar_checkout_id)
  where polar_checkout_id is not null;

comment on column public.profiles.polar_checkout_id is 'Polar checkout session used at signup.';
comment on column public.profiles.polar_customer_id is 'Polar customer id from completed checkout.';
comment on column public.profiles.subscription_status is 'e.g. active — set when Phase One payment is confirmed.';
