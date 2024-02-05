use crate::IntoRequest;
use derive_builder::Builder;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Clone, Debug, Builder)]
pub struct ChatCompletionRequest {
    #[builder(setter(into))]
    messages: Vec<ChatCompletionMessage>,
    #[builder(default, setter(into))]
    model: ChatCompletionModel,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<i32>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<i32>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormatObject>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<usize>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[builder(default, setter(into))]
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<Tool>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[builder(default, setter(strip_option, into))]
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Serialize, Clone, Debug, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    None,
    Auto,
    Function {
        name: String,
        r#type: ToolCallType,
    },
}

#[derive(Serialize, Clone, Debug)]
pub struct Tool {
    r#type: ToolCallType,
    function: FunctionInfo,
}

#[derive(Serialize, Clone, Debug)]
pub struct FunctionInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    name: String,
    parameters: serde_json::Value,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub struct ResponseFormatObject {
    r#type: ResponseFormat,
}

#[derive(Serialize, Clone, Debug, Copy, Default)]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    #[default]
    Json,
}

#[derive(Serialize, Clone, Debug, Copy, Default)]
pub enum ChatCompletionModel {
    #[serde(rename = "gpt-3.5-turbo")]
    #[default]
    Gpt3Dot5Turbo,
}

#[derive(Serialize, Clone, Debug)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatCompletionMessage {
    User(UserMessage),
    Assistant(AssistantMessage),
    System(SystemMessage),
    Tool(ToolMessage),
}

#[derive(Serialize, Clone, Debug)]
pub struct UserMessage {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct AssistantMessage {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<ToolCall>,
}

#[derive(Serialize, Clone, Debug)]
pub struct SystemMessage {
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct ToolMessage {
    content: String,
    tool_call_id: String,
}

#[derive(Serialize, Debug, Clone)]
pub struct ToolCall {
    id: String,
    r#type: ToolCallType,
    function: FunctionCall,
}

#[derive(Serialize, Debug, Clone, Copy, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallType {
    #[default]
    Function,
}

#[derive(Serialize, Debug, Clone)]
pub struct FunctionCall {
    name: String,
    arguments: String,
}

impl IntoRequest for ChatCompletionRequest {
    fn into_request(self, client: Client) -> RequestBuilder {
        client.post("https://api.openai.com/v1/chat/completions")
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: ChatCompletionUsage,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ChatCompletionChoice {
    pub index: i32,
    pub finish_reason: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

impl ChatCompletionMessage {
    pub fn new_user_message(content: impl Into<String>, name: &str) -> Self {
        Self::User(UserMessage {
            content: content.into(),
            name: Self::get_name(name),
        })
    }

    pub fn new_assistant_message(content: impl Into<String>, name: &str) -> Self {
        Self::Assistant(AssistantMessage {
            content: content.into(),
            name: Self::get_name(name),
            tool_calls: vec![],
        })
    }

    pub fn new_system_message(content: impl Into<String>, name: &str) -> Self {
        Self::System(SystemMessage {
            content: content.into(),
            name: Self::get_name(name),
        })
    }

    fn get_name(name: &str) -> Option<String> {
        if name.is_empty() {
            None
        } else {
            Some(name.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_completion_request_should_serialize() {
        let messages = vec![
            ChatCompletionMessage::new_user_message("test user", "test"),
            ChatCompletionMessage::new_assistant_message("test ass", "test"),
            ChatCompletionMessage::new_system_message("test sys", "test"),
        ];
        let req = ChatCompletionRequestBuilder::default()
            .model(ChatCompletionModel::Gpt3Dot5Turbo)
            .messages(messages)
            .build()
            .unwrap();
        assert_eq!(
            serde_json::to_value(req).unwrap(),
            json!({
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {
                                "role": "user",
                                "content": "test user",
                                "name": "test",
                            },
                            {
                                "role": "assistant",
                                "content": "test ass",
                                "name": "test",
                            }, {
                                "role": "system",
                                "content": "test sys",
                                "name": "test",
                            }]
            })
        );
    }
}
