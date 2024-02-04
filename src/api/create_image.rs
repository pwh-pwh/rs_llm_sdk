use crate::IntoRequest;
use derive_builder::Builder;
use reqwest::{Client, RequestBuilder};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Clone, Debug, Builder)]
#[builder(pattern = "mutable")]
pub struct CreateImageRequest {
    #[builder(setter(into))]
    pub prompt: String,
    #[builder(default)]
    model: ImageModel,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<ImageQuality>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ImageResponseFormat>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<ImageSize>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<ImageStyle>,
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Serialize, Clone, Debug, Eq, PartialEq, Default)]
pub enum ImageModel {
    #[serde(rename = "dall-e-3")]
    #[default]
    DallE3,
}

#[derive(Serialize, Clone, Debug, Eq, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageQuality {
    #[default]
    Standard,
    Hd,
}

#[derive(Serialize, Clone, Debug, Eq, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageResponseFormat {
    #[default]
    Url,
    B64Json,
}

#[derive(Serialize, Clone, Debug, Eq, PartialEq, Default)]
pub enum ImageSize {
    #[serde(rename = "1024x1024")]
    #[default]
    Large,
    #[serde(rename = "1792x1024")]
    LargeWide,
    #[serde(rename = "1024x1792")]
    LargeTail,
}

#[derive(Serialize, Clone, Debug, Eq, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageStyle {
    #[default]
    Vivid,
    Natural,
}

impl IntoRequest for CreateImageRequest {
    fn into_request(self, client: Client) -> RequestBuilder {
        client
            .post("https://api.openai.com/v1/images/generations")
            .json(&self)
    }
}

impl CreateImageRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        CreateImageRequestBuilder::default()
            .prompt(prompt)
            .build()
            .unwrap()
    }
}

#[derive(Deserialize, Clone, Debug)]
pub struct CreateImageResponse {
    pub created: u64,
    pub data: Vec<ImageObject>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct ImageObject {
    /// The base64-encoded JSON of the generated image, if response_format is b64_json.
    pub b64_json: Option<String>,
    pub url: Option<String>,
    /// revised_prompt
    pub revised_prompt: String,
}

//测试模块
#[cfg(test)]
mod test {
    use super::*;
    use crate::LlmSdk;
    use anyhow::Result;
    use serde_json::json;
    use std::env;

    #[test]
    fn create_image_request_should_serialize() -> Result<()> {
        let request = CreateImageRequestBuilder::default()
            .prompt("test")
            .build()
            .unwrap();
        assert_eq!(
            serde_json::to_value(request)?,
            json!({
                "prompt": "test",
                "model": "dall-e-3",
            })
        );
        Ok(())
    }

    #[test]
    fn create_image_request_should_custom_serialize() -> Result<()> {
        let request = CreateImageRequestBuilder::default()
            .prompt("test")
            .style(ImageStyle::Natural)
            .quality(ImageQuality::Hd)
            .build()
            .unwrap();
        assert_eq!(
            serde_json::to_value(request)?,
            json!({
                "prompt": "test",
                "model": "dall-e-3",
                "style": "natural",
                "quality": "hd",
            })
        );
        Ok(())
    }

    #[tokio::test]
    async fn create_image_should_work() -> Result<()> {
        let client = LlmSdk::new(env::var("OPENAI_API_KEY").unwrap());
        let request = CreateImageRequest::new("test");
        let response = client.create_image(request).await?;
        assert_eq!(response.data.len(), 1);
        let image = &response.data[0];
        assert!(image.b64_json.is_none());
        assert!(image.url.is_some());
        println!("{image:?}");
        //write image to file
        let _ = std::fs::write(
            "image.png",
            reqwest::get(image.url.as_ref().unwrap())
                .await?
                .bytes()
                .await?,
        )?;
        Ok(())
    }
}
