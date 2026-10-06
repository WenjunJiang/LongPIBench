TEXT = ("Hi, I'm Daniel Whitfield, born 14 March 1987. Reach me at daniel.w@meridiancap.com "
        "or +1 415-555-0123; my account is 4485-2210-9931-0042 and I live at 221B Baker Street, London.")

def param_count(m):
    return sum(p.numel() for p in m.parameters())

def banner(s):
    print("\n" + "=" * 8 + " " + s + " " + "=" * 8)
