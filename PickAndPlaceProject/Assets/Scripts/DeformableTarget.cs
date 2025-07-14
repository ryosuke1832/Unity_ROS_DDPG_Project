using UnityEngine;
using System.Collections.Generic;

public class DeformableTarget : MonoBehaviour
{
    [Header("変形設定")]
    [Range(0f, 1f)]
    public float softness = 0.5f;
    public float maxDeformation = 0.3f;
    public float deformationSpeed = 2f;
    public bool enableVisualDeformation = true;
    
    [Header("物理特性")]
    public float breakingForce = 50f;
    public float compressionResistance = 15f;
    
    [Header("視覚設定")]
    public Color originalColor = Color.white;
    
    [Header("デバッグ")]
    public bool enableDebugLogs = false;
    
    // 内部状態
    private Vector3 originalScale;
    private float currentDeformation = 0f;
    private bool isBroken = false;
    private Renderer objectRenderer;
    private bool isBeingGrasped = false;
    private float appliedForce = 0f;
    
    void Start()
    {
        originalScale = transform.localScale;
        objectRenderer = GetComponent<Renderer>();
        if (objectRenderer != null)
            originalColor = objectRenderer.material.color;
        
        Debug.Log("DeformableTarget initialized");
    }
    
    void Update()
    {
        UpdateDeformation();
        UpdateVisualFeedback();
    }
    
    public void ApplyGripperForce(float force, Vector3 contactPosition)
    {
        if (isBroken) return;
        
        appliedForce = force;
        isBeingGrasped = force > 0.1f;
        
        if (force > breakingForce)
        {
            isBroken = true;
            Debug.Log($"Object broken! Force: {force:F2}N");
        }
        
        CalculateDeformation(force);
        
        if (enableDebugLogs)
            Debug.Log($"Force applied: {force:F2}N, Deformation: {currentDeformation:F3}");
    }
    
    public void ApplyGripperForceWithDirection(float force, Vector3 contactPosition, Vector3 contactNormal)
    {
        if (isBroken) return;
        
        appliedForce = force;
        isBeingGrasped = force > 0.1f;
        
        if (force > breakingForce)
        {
            isBroken = true;
            Debug.Log($"Object broken! Force: {force:F2}N");
        }
        
        CalculateDirectionalDeformation(force, contactNormal);
        
        if (enableDebugLogs)
            Debug.Log($"Directional force applied: {force:F2}N, Normal: {contactNormal}");
    }
    
    private void CalculateDeformation(float force)
    {
        float targetDeformation = Mathf.Clamp01(force / compressionResistance) * softness;
        targetDeformation = Mathf.Clamp(targetDeformation, 0f, maxDeformation);
        currentDeformation = Mathf.Lerp(currentDeformation, targetDeformation, deformationSpeed * Time.deltaTime);
    }
    
    private void CalculateDirectionalDeformation(float force, Vector3 contactNormal)
    {
        float baseDeformation = Mathf.Clamp01(force / compressionResistance) * softness;
        float directionFactor = Mathf.Abs(Vector3.Dot(contactNormal, Vector3.up));
        directionFactor = Mathf.Clamp(directionFactor, 0.3f, 1f);
        
        float targetDeformation = baseDeformation * directionFactor;
        targetDeformation = Mathf.Clamp(targetDeformation, 0f, maxDeformation);
        currentDeformation = Mathf.Lerp(currentDeformation, targetDeformation, deformationSpeed * Time.deltaTime);
    }
    
    private void UpdateDeformation()
    {
        if (!enableVisualDeformation || isBroken) return;
        
        if (!isBeingGrasped)
        {
            currentDeformation = Mathf.Lerp(currentDeformation, 0f, deformationSpeed * Time.deltaTime);
        }
        
        float compressionFactor = 1f - currentDeformation;
        Vector3 deformedScale = new Vector3(
            originalScale.x * (1f + currentDeformation * 0.2f),
            originalScale.y * compressionFactor,
            originalScale.z * (1f + currentDeformation * 0.2f)
        );
        
        transform.localScale = Vector3.Lerp(transform.localScale, deformedScale, Time.deltaTime * deformationSpeed);
    }
    
    private void UpdateVisualFeedback()
    {
        if (objectRenderer == null) return;
        
        Color stressColor = Color.Lerp(originalColor, Color.red, currentDeformation);
        objectRenderer.material.color = stressColor;
        
        if (isBroken)
        {
            objectRenderer.material.color = Color.gray;
        }
    }
    
    public ObjectState GetCurrentState()
    {
        return new ObjectState
        {
            deformation = currentDeformation,
            appliedForce = appliedForce,
            isBroken = isBroken,
            isBeingGrasped = isBeingGrasped,
            materialType = 1, // Medium
            softness = softness
        };
    }
    
    public void ResetObject()
    {
        isBroken = false;
        currentDeformation = 0f;
        appliedForce = 0f;
        isBeingGrasped = false;
        transform.localScale = originalScale;
        
        if (objectRenderer != null)
            objectRenderer.material.color = originalColor;
        
        Debug.Log("Object reset");
    }
}