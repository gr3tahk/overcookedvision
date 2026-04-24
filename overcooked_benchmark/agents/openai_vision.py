from __future__ import annotations

from overcooked_benchmark.agents.base import ACTION_TO_OVERCOOKED, AgentDecision, AgentObservation, BenchmarkAgent, parse_agent_response
from overcooked_benchmark.agents.prompts import build_action_prompt
from overcooked_benchmark.openai_client import create_chat_completion
from overcooked_benchmark.rendering import render_state_data_url


class OpenAIVisionAgent(BenchmarkAgent):
    def __init__(self, player_id: int, player_name: str, client, model: str):
        super().__init__(player_id, player_name)
        self.client = client
        self.model = model

    def act(self, observation: AgentObservation):
        prompt = build_action_prompt(observation, include_text_state=False)
        image_url = render_state_data_url(observation.state, observation.mdp, tick=observation.tick, score=observation.score)
        response = create_chat_completion(
            self.client,
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=120,
        )
        raw = response.choices[0].message.content or ""
        action, message, plan, valid, invalid_reason = parse_agent_response(raw)
        self.last_decision = AgentDecision(
            player_id=self.player_id,
            player_name=self.player_name,
            action=action,
            message=message,
            plan=plan,
            raw_response=raw,
            prompt=prompt,
            valid=valid,
            invalid_reason=invalid_reason,
        )
        return ACTION_TO_OVERCOOKED[action]
