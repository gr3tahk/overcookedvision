from benchmark import run_game

print("Running full 400 tick game on cramped_room...")
score = run_game('cramped_room')
print(f"\nFinal score: {score}")
print(f"Soups delivered: {score // 20}")
